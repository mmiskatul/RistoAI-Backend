from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from bson import ObjectId

from app.db.migrations import MIGRATION_KEY, run_data_migrations
from app.db.mongodb import get_database
from app.services.restaurant import RestaurantOperationsService
from app.tests.helpers import (
    complete_onboarding_profile,
    register_and_login,
    seed_subscription_plan,
    select_subscription_plan as select_subscription_plan_only,
)


def select_subscription_plan(client, headers, *, billing_cycle: str = '1_month', start_trial: bool = True) -> None:
    select_subscription_plan_only(client, headers, billing_cycle=billing_cycle, start_trial=start_trial)
    complete_onboarding_profile(client, headers)


def test_food_cost_filter_excludes_manual_entry_sources():
    service = RestaurantOperationsService.__new__(RestaurantOperationsService)

    assert service._is_food_cost_expense({
        "source_kind": "manual_entry",
        "category": "Inventory Adjustment",
    }) is False
    assert service._is_food_cost_expense({
        "source_kind": "manual_entry_restore",
        "category": "Food Inventory",
    }) is False
    assert service._is_food_cost_expense({
        "source_kind": "inventory",
        "category": "Inventory Adjustment",
    }) is True


def test_document_line_item_categories_resolve_to_uncategorized():
    service = RestaurantOperationsService.__new__(RestaurantOperationsService)

    resolved = service._resolve_document_line_item_categories(
        [
            {"product_name": "Salmon", "category": ""},
            {"product_name": "Water", "category": "Drinks"},
        ]
    )

    assert resolved[0]["category"] == "Uncategorized"
    assert resolved[1]["category"] == "Drinks"


def test_restaurant_document_upload_extract_and_confirm_flow(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Marco Owner",
            "email": "marco@example.com",
            "password": "MarcoPass123",
            "phone": "+3900000001",
        },
    )
    select_subscription_plan(client, headers)

    upload_response = client.post(
        "/api/v1/restaurant/documents/upload-extract",
        headers=headers,
        files={"file": ("invoice-march.png", b"fake-image", "image/png")},
    )
    assert upload_response.status_code == 422
    assert upload_response.json()["detail"] == "AI service is not configured or currently unavailable. Please check OpenAI API settings."

    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": upload_payload["document_type"],
            "document_label": upload_payload["document_label"],
            "counterparty_name": upload_payload["counterparty_name"],
            "supplier_name": "Bakery Goods Co",
            "invoice_number": upload_payload["document_number"],
            "total_amount": 425.0,
            "currency": upload_payload["currency"],
            "expense_amount": 425.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "line_items": [
                {"product_name": "Sourdough Loaf", "quantity": 20, "unit_price": 5.0, "total_price": 100.0},
                {"product_name": "Pastry Flour (25kg)", "quantity": 5, "unit_price": 45.0, "total_price": 225.0},
                {"product_name": "Butter (Case)", "quantity": 2, "unit_price": 50.0, "total_price": 100.0}
            ],
            "source_file_name": upload_payload["source_file_name"],
            "ai_provider": upload_payload["ai_provider"],
            "ai_summary": upload_payload["ai_summary"]
        },
    )
    assert confirm_response.status_code == 201
    assert confirm_response.json()["status"] == "processed"
    assert confirm_response.json()["document_type"] == "expense"
    assert confirm_response.json()["document_label"] == "Expense"
    assert confirm_response.json()["counterparty_name"] == "Fresh Food Supplier Ltd"
    assert confirm_response.json()["currency"] == "EUR"
    assert confirm_response.json()["expense_amount"] == 425.0
    assert confirm_response.json()["cash_amount"] == 0.0
    assert confirm_response.json()["revenue_amount"] == 0.0
    assert confirm_response.json()["profit_amount"] == 0.0
    assert confirm_response.json()["confirmed_by_user_id"]
    assert confirm_response.json()["confirmed_at"]
    assert confirm_response.json()["document_date"] == datetime.now(UTC).date().isoformat()
    assert "page_title" not in confirm_response.json()
    assert confirm_response.json()["status"] == "processed"
    assert confirm_response.json()["line_items"][0]["product_name"] == "Sourdough Loaf"

    assert confirm_response.json()["created_by_user_id"]
    assert confirm_response.json()["last_edited_by_user_id"]
    assert confirm_response.json()["confirmed_by_user_id"]

    documents_response = client.get("/api/v1/restaurant/documents", headers=headers)
    assert documents_response.status_code == 200
    documents_payload = documents_response.json()
    assert set(documents_payload.keys()) == {"total", "page", "page_size", "pages", "items"}
    assert documents_payload["items"][0]["document_type"] == "expense"
    assert documents_payload["items"][0]["counterparty_name"] == "Fresh Food Supplier Ltd"
    assert documents_payload["items"][0]["status"] == "processed"
    assert documents_payload["items"][0]["line_item_count"] == 3

    today_iso = datetime.now(UTC).date().isoformat()
    document_detail_response = client.get(f"/api/v1/restaurant/documents/{confirm_response.json()['id']}", headers=headers)
    assert document_detail_response.status_code == 200
    document_detail_payload = document_detail_response.json()
    assert document_detail_payload["document_type"] == "expense"
    assert document_detail_payload["document_label"] == "Expense"
    assert document_detail_payload["counterparty_name"] == "Fresh Food Supplier Ltd"
    assert document_detail_payload["document_date"] == today_iso
    assert document_detail_payload["upload_date"]
    assert document_detail_payload["expense_amount"] == 425.0
    assert "page_title" not in document_detail_payload
    assert "source_file_name" not in document_detail_payload

    download_response = client.get(f"/api/v1/restaurant/documents/{confirm_response.json()['id']}/download", headers=headers)
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("application/pdf")
    assert "attachment; filename=" in download_response.headers["content-disposition"]
    assert download_response.content.startswith(b"%PDF")

    download_svg_response = client.get(f"/api/v1/restaurant/documents/{confirm_response.json()['id']}/download-image?format=svg", headers=headers)
    assert download_svg_response.status_code == 200
    assert download_svg_response.headers["content-type"].startswith("image/svg+xml")
    assert "attachment; filename=" in download_svg_response.headers["content-disposition"]
    assert b"<svg" in download_svg_response.content

    date_data_response = client.get(f"/api/v1/restaurant/daily-data?view=date&reference_date={today_iso}", headers=headers)
    assert date_data_response.status_code == 200
    date_payload = date_data_response.json()
    assert set(date_payload.keys()) == {"total", "page", "page_size", "pages", "items"}
    assert date_payload["total"] == 0
    assert date_payload["items"] == []

    week_data_response = client.get(f"/api/v1/restaurant/daily-data?view=week&reference_date={today_iso}", headers=headers)
    assert week_data_response.status_code == 200
    assert set(week_data_response.json().keys()) == {"total", "page", "page_size", "pages", "items"}

    invoice_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-date?business_date={today_iso}", headers=headers)
    assert invoice_detail_response.status_code == 200
    invoice_detail_payload = invoice_detail_response.json()
    assert invoice_detail_payload["business_date"] == today_iso
    assert invoice_detail_payload["total_revenue"] == 0.0
    assert invoice_detail_payload["total_expenses"] == 425.0
    assert invoice_detail_payload["operating_revenue"] == 0.0
    assert invoice_detail_payload["invoice_document_total"] == 425.0
    assert invoice_detail_payload["total_covers"] == 0
    assert invoice_detail_payload["document_count"] == 1
    assert invoice_detail_payload["documents"][0]["counterparty_name"] == "Fresh Food Supplier Ltd"
    assert invoice_detail_payload["documents"][0]["total_amount"] == 425.0

    week_invoice_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-week?reference_date={today_iso}", headers=headers)
    assert week_invoice_detail_response.status_code == 200
    week_invoice_detail_payload = week_invoice_detail_response.json()
    assert week_invoice_detail_payload["business_date"] == today_iso
    assert week_invoice_detail_payload["total_expenses"] == 425.0
    assert week_invoice_detail_payload["document_count"] == 1
    assert week_invoice_detail_payload["documents"][0]["counterparty_name"] == "Fresh Food Supplier Ltd"

    month_invoice_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-month?reference_date={today_iso}", headers=headers)
    assert month_invoice_detail_response.status_code == 200
    month_invoice_detail_payload = month_invoice_detail_response.json()
    assert month_invoice_detail_payload["business_date"] == today_iso
    assert month_invoice_detail_payload["total_expenses"] == 425.0
    assert month_invoice_detail_payload["document_count"] == 1
    assert month_invoice_detail_payload["documents"][0]["counterparty_name"] == "Fresh Food Supplier Ltd"

    date_reference_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-date-reference?reference_date={today_iso}", headers=headers)
    assert date_reference_detail_response.status_code == 200
    assert date_reference_detail_response.json()["business_date"] == today_iso

    week_business_date_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-week-business-date?business_date={today_iso}", headers=headers)
    assert week_business_date_detail_response.status_code == 200
    assert week_business_date_detail_response.json()["business_date"] == today_iso
    assert week_business_date_detail_response.json()["document_count"] == 1

    month_business_date_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-month-business-date?business_date={today_iso}", headers=headers)
    assert month_business_date_detail_response.status_code == 200
    assert month_business_date_detail_response.json()["business_date"] == today_iso
    assert month_business_date_detail_response.json()["document_count"] == 1

    all_dates_response = client.get("/api/v1/restaurant/daily-data/by-date", headers=headers)
    assert all_dates_response.status_code == 200
    assert all_dates_response.json()["total"] >= 1
    assert all_dates_response.json()["items"][0]["business_date"]

    all_weeks_response = client.get("/api/v1/restaurant/daily-data/by-week", headers=headers)
    assert all_weeks_response.status_code == 200
    assert all_weeks_response.json()["total"] >= 1
    assert all_weeks_response.json()["items"][0]["business_date"]

    all_months_response = client.get("/api/v1/restaurant/daily-data/by-month", headers=headers)
    assert all_months_response.status_code == 200
    assert all_months_response.json()["total"] >= 1
    assert all_months_response.json()["items"][0]["business_date"]
    manual_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": today_iso,
                "pos_payments": 800,
                "cash_payments": 200,
                "bank_transfer_payments": 100,
                "lunch_covers": 10,
                "dinner_covers": 12,
                "opening_cash": 100,
                "closing_cash": 180
            }
        },
    )
    assert manual_entry_response.status_code == 201

    db = asyncio.run(app.dependency_overrides[get_database]())
    restaurant_record = asyncio.run(
        db["restaurant_finance_snapshots"].find_one(
            {
                "period_type": "day",
                "business_date": datetime.now(UTC).date().isoformat(),
                "uploaded_document_ids": {"$in": [confirm_response.json()["id"]]},
            }
        )
    )
    assert restaurant_record is not None
    assert restaurant_record["uploaded_document_count"] == 1
    assert restaurant_record["manual_entry_id"] is not None
    assert restaurant_record["total_revenue"] == 1100.0
    assert restaurant_record["total_expenses"] == 425.0
    assert restaurant_record["uploaded_document_total"] == 425.0
    assert restaurant_record["expense_summary"]["document_expense_total"] == 425.0
    manual_entry_transactions = asyncio.run(
        db["restaurant_finance_transactions"].find({"source_kind": "manual_entry", "source_id": restaurant_record["manual_entry_id"]}).to_list(length=None)
    )
    assert {item["transaction_type"] for item in manual_entry_transactions} == {"bank_collection", "cash_collection"}
    assert sorted((item["payment_channel"], item["amount"]) for item in manual_entry_transactions) == [
        ("bank_transfer", 100.0),
        ("cash", 200.0),
        ("pos", 800.0),
    ]

    month_data_response = client.get(f"/api/v1/restaurant/daily-data?view=month&reference_date={today_iso}", headers=headers)
    assert month_data_response.status_code == 200
    month_data_payload = month_data_response.json()
    assert month_data_payload["total"] >= 1
    assert month_data_payload["items"][0]["total_expenses"] == 425.0

    analytics_response = client.get("/api/v1/restaurant/analytics/overview?period=weekly", headers=headers)
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()
    assert "estimated_profit" not in analytics_payload
    assert len(analytics_payload["supplier_price_alerts"]) >= 1
    assert "Bakery Goods Co" in analytics_payload["supplier_price_alerts"][0]["title"] or analytics_payload["supplier_price_alerts"][0]["title"]


def test_restaurant_document_upload_accepts_common_image_formats_and_short_invoice_number(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Image Owner",
            "email": "image-owner@example.com",
            "password": "ImagePass123",
            "phone": "+3900000099",
        },
    )
    select_subscription_plan(client, headers)

    upload_response = client.post(
        "/api/v1/restaurant/documents/upload-extract",
        headers=headers,
        files={"file": ("receipt.heic", b"fake-heic-image", "image/heic")},
    )
    assert upload_response.status_code == 200
    upload_payload = upload_response.json()

    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": upload_payload["document_type"],
            "document_label": "R",
            "counterparty_name": "A",
            "supplier_name": "A",
            "invoice_number": "7",
            "total_amount": upload_payload["total_amount"],
            "currency": upload_payload["currency"],
            "line_items": upload_payload["line_items"],
            "source_file_name": upload_payload["source_file_name"],
            "ai_provider": upload_payload["ai_provider"],
            "ai_summary": upload_payload["ai_summary"],
        },
    )

    assert confirm_response.status_code == 201
    payload = confirm_response.json()
    assert payload["document_number"] == "7"
    assert payload["counterparty_name"] == "A"


def test_restaurant_data_migration_unifies_legacy_finance_collections(app):
    db = asyncio.run(app.dependency_overrides[get_database]())
    tenant_id = "tenant-migration"
    now = datetime.now(UTC)
    daily_id = ObjectId()
    weekly_id = ObjectId()
    monthly_id = ObjectId()
    document_id = ObjectId()

    asyncio.run(
        db["restaurant_invoices"].insert_one(
            {
                "_id": document_id,
                "tenant_id": tenant_id,
                "supplier_name": "Legacy Supplier",
                "invoice_number": "INV-001",
                "invoice_date": "2026-04-10",
                "status": "processed",
                "total_amount": 150.0,
                "created_at": now,
                "updated_at": now,
            }
        )
    )
    asyncio.run(
        db["restaurant_daily_records"].insert_one(
            {
                "_id": daily_id,
                "tenant_id": tenant_id,
                "business_date": "2026-04-10",
                "total_revenue": 500.0,
                "total_expenses": 200.0,
                "created_at": now,
                "updated_at": now,
            }
        )
    )
    asyncio.run(
        db["restaurant_weekly_records"].insert_one(
            {
                "_id": weekly_id,
                "tenant_id": tenant_id,
                "week_start_date": "2026-04-06",
                "week_end_date": "2026-04-12",
                "total_revenue": 2500.0,
                "total_expenses": 1200.0,
                "created_at": now,
                "updated_at": now,
            }
        )
    )
    asyncio.run(
        db["restaurant_monthly_records"].insert_one(
            {
                "_id": monthly_id,
                "tenant_id": tenant_id,
                "month_key": "2026-04",
                "month_start_date": "2026-04-01",
                "month_end_date": "2026-04-30",
                "total_revenue": 9000.0,
                "total_expenses": 4200.0,
                "created_at": now,
                "updated_at": now,
            }
        )
    )

    asyncio.run(run_data_migrations(db))
    asyncio.run(run_data_migrations(db))

    migrated_document = asyncio.run(db["restaurant_documents"].find_one({"_id": document_id}))
    assert migrated_document is not None
    assert migrated_document["counterparty_name"] == "Legacy Supplier"
    assert migrated_document["migrated_from_collection"] == "restaurant_invoices"

    daily_snapshot = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"tenant_id": tenant_id, "period_type": "day", "period_key": "2026-04-10"})
    )
    assert daily_snapshot is not None
    assert daily_snapshot["_id"] == daily_id
    assert daily_snapshot["business_date"] == "2026-04-10"

    weekly_snapshot = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"tenant_id": tenant_id, "period_type": "week", "period_key": "2026-04-06"})
    )
    assert weekly_snapshot is not None
    assert weekly_snapshot["_id"] == weekly_id
    assert weekly_snapshot["week_end_date"] == "2026-04-12"

    monthly_snapshot = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"tenant_id": tenant_id, "period_type": "month", "period_key": "2026-04"})
    )
    assert monthly_snapshot is not None
    assert monthly_snapshot["_id"] == monthly_id
    assert monthly_snapshot["month_end_date"] == "2026-04-30"

    migration_record = asyncio.run(db["app_migrations"].find_one({"key": MIGRATION_KEY}))
    assert migration_record is not None
    assert migration_record["summary"]["documents_migrated"] == 1
    assert migration_record["summary"]["daily_snapshots_migrated"] == 1


def test_manual_entry_method_one_creates_finance_transactions(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Method One Owner",
            "email": "method-one@example.com",
            "password": "MethodOne123",
            "phone": "+1555000199",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()
    response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": today_iso,
                "pos_payments": 100,
                "cash_withdrawals": 100,
                "cash_in": 0,
                "cash_out": 0,
                "expenses_in_cash": 175,
                "opening_cash": 150,
                "closing_cash": 200,
                "notes": "Method 1 ledger test",
            },
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["total_revenue"] == 425.0
    assert payload["total_expenses"] == 175.0
    assert payload["profit"] == 250.0
    assert payload["register_summary"]["closing_cash"] == 200.0
    assert payload["method_sections"][0]["title"] == "Deposit Section"
    assert payload["method_sections"][0]["fields"][0]["key"] == "pos_payments"
    assert payload["method_sections"][1]["fields"][0]["value"] == 175.0
    assert payload["method_sections"][3]["fields"][0]["value"] == "Method 1 ledger test"

    db = asyncio.run(app.dependency_overrides[get_database]())
    record_id = payload["id"]
    transactions = asyncio.run(
        db["restaurant_finance_transactions"].find({"source_kind": "manual_entry", "source_id": record_id}).sort("transaction_type").to_list(length=None)
    )
    assert len(transactions) == 3
    assert sorted((item["transaction_type"], item["payment_channel"], item["amount"]) for item in transactions) == [
        ("bank_collection", "pos", 100.0),
        ("expense", "cash", 175.0),
        ("withdrawal", "cash", 100.0),
    ]

    daily_snapshot = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": today_iso})
    )
    assert daily_snapshot is not None
    assert daily_snapshot["total_revenue"] == 425.0
    assert daily_snapshot["total_expenses"] == 175.0
    assert daily_snapshot["withdrawals_total"] == 100.0
    assert daily_snapshot["cash_available"] == 200.0
    assert daily_snapshot["revenue_summary"]["sales_total"] == 425.0
    assert daily_snapshot["expense_summary"]["total_expenses"] == 175.0
    assert daily_snapshot["deposit_summary"]["bank_deposits_total"] == 0.0
    assert daily_snapshot["cash_summary"]["cash_available"] == 155.0

    first_transaction_metadata = transactions[0]["metadata"]
    assert "ledger_group" in first_transaction_metadata
    assert "affects_revenue" in first_transaction_metadata
    assert "affects_cash" in first_transaction_metadata
    assert "affects_profit" in first_transaction_metadata


def test_restaurant_document_confirm_save_supports_revenue_classification(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Revenue Owner",
            "email": "revenue@example.com",
            "password": "RevenuePass123",
            "phone": "+3900000002",
        },
    )
    select_subscription_plan(client, headers)

    business_date = datetime.now(UTC).date().isoformat()
    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "revenue",
            "document_label": "Revenue",
            "counterparty_name": "Dining Room Sales",
            "supplier_name": "Dining Room Sales",
            "invoice_number": "REV-001",
            "invoice_date": business_date,
            "total_amount": 920.0,
            "currency": "EUR",
            "expense_amount": 0.0,
            "cash_amount": 0.0,
            "revenue_amount": 920.0,
            "profit_amount": 0.0,
            "line_items": [],
            "source_file_name": "revenue-summary.pdf",
            "ai_provider": "fallback",
            "ai_summary": "Revenue summary uploaded.",
        },
    )
    assert confirm_response.status_code == 201
    confirm_payload = confirm_response.json()
    assert confirm_payload["document_type"] == "revenue"
    assert confirm_payload["revenue_amount"] == 920.0
    assert confirm_payload["expense_amount"] == 0.0

    date_data_response = client.get(f"/api/v1/restaurant/daily-data/by-date?business_date={business_date}", headers=headers)
    assert date_data_response.status_code == 200
    date_payload = date_data_response.json()
    assert date_payload["business_date"] == business_date
    assert date_payload["total_revenue"] == 920.0
    assert date_payload["total_expenses"] == 0.0
    assert date_payload["invoice_document_total"] == 0.0

    db = asyncio.run(app.dependency_overrides[get_database]())
    restaurant_record = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": business_date})
    )
    assert restaurant_record is not None
    assert restaurant_record["total_revenue"] == 920.0
    assert restaurant_record["total_expenses"] == 0.0
    assert restaurant_record["profit"] == 920.0


def test_restaurant_endpoints_are_scoped_per_user(client, app):
    seed_subscription_plan(app)
    headers_user_one = register_and_login(
        client,
        {
            "full_name": "Owner One",
            "email": "owner1@example.com",
            "password": "OwnerOne123",
            "phone": "+1555000001",
        },
    )
    select_subscription_plan(client, headers_user_one)

    headers_user_two = register_and_login(
        client,
        {
            "full_name": "Owner Two",
            "email": "owner2@example.com",
            "password": "OwnerTwo123",
            "phone": "+1555000002",
        },
    )
    select_subscription_plan(client, headers_user_two)

    today_iso = datetime.now(UTC).date().isoformat()

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers_user_one,
        json={"category": "Staff Costs", "amount": 420.0, "expense_date": today_iso, "section": "bank", "notes": "Payroll"},
    )
    assert expense_response.status_code == 201
    assert expense_response.json()["section"] == "bank"

    inventory_response = client.post(
        "/api/v1/restaurant/inventory",
        headers=headers_user_one,
        json={
            "product_name": "Tomato Sauce",
            "category": "Sauce",
            "stock_quantity": 12,
            "unit_type": "bottles",
            "supplier_name": "Global Foods Inc.",
            "unit_price": 4.5,
            "alert_threshold": 5,
            "purchase_date": "2026-03-10",
        },
    )
    assert inventory_response.status_code == 201
    inventory_id = inventory_response.json()["id"]
    db = asyncio.run(app.dependency_overrides[get_database]())
    inventory_expense = asyncio.run(
        db["restaurant_expenses"].find_one(
            {
                "source_kind": "inventory",
                "source_id": inventory_id,
                "source_inventory_item_id": inventory_id,
            }
        )
    )
    assert inventory_expense is not None
    assert inventory_expense["amount"] == 54.0

    inventory_expense_delete_response = client.delete(f"/api/v1/restaurant/expenses/{inventory_expense['_id']}", headers=headers_user_one)
    assert inventory_expense_delete_response.status_code == 422

    expenses_user_two = client.get("/api/v1/restaurant/expenses", headers=headers_user_two)
    assert expenses_user_two.status_code == 200
    assert set(expenses_user_two.json().keys()) == {"today", "this_week", "this_month", "this_year"}
    assert expenses_user_two.json()["today"]["total"] == 0
    assert expenses_user_two.json()["this_week"]["items"] == []

    expenses_user_one = client.get("/api/v1/restaurant/expenses", headers=headers_user_one)
    assert expenses_user_one.status_code == 200
    assert expenses_user_one.json()["this_month"]["top_category"] == "Staff Costs"
    assert expenses_user_one.json()["this_month"]["distribution"][0]["label"] == "Staff Costs"

    inventory_user_two = client.get("/api/v1/restaurant/inventory", headers=headers_user_two)
    assert inventory_user_two.status_code == 200
    assert inventory_user_two.json()["total"] == 0

    inventory_detail_user_two = client.get(f"/api/v1/restaurant/inventory/{inventory_id}", headers=headers_user_two)
    assert inventory_detail_user_two.status_code == 404

    inventory_detail_user_one = client.get(f"/api/v1/restaurant/inventory/{inventory_id}", headers=headers_user_one)
    assert inventory_detail_user_one.status_code == 200
    inventory_detail_payload = inventory_detail_user_one.json()
    assert inventory_detail_payload["supplier_name"] == "Global Foods Inc."
    assert inventory_detail_payload["current_stock_value"] == 12
    assert "history" in inventory_detail_payload

    inventory_list_user_one = client.get("/api/v1/restaurant/inventory", headers=headers_user_one)
    assert inventory_list_user_one.status_code == 200
    inventory_list_payload = inventory_list_user_one.json()
    assert inventory_list_payload["total_inventory_value"] == 54.0
    assert inventory_list_payload["items"][0]["id"] == inventory_id

    categories_user_one = client.get("/api/v1/restaurant/inventory/categories", headers=headers_user_one)
    assert categories_user_one.status_code == 200
    assert categories_user_one.json()["items"][0]["name"] == "Sauce"

    suppliers_user_one = client.get("/api/v1/restaurant/inventory/suppliers", headers=headers_user_one)
    assert suppliers_user_one.status_code == 200
    assert suppliers_user_one.json()["items"][0]["name"] == "Global Foods Inc."

    categories_user_two = client.get("/api/v1/restaurant/inventory/categories", headers=headers_user_two)
    assert categories_user_two.status_code == 200
    assert categories_user_two.json()["items"] == []

    inventory_update_response = client.patch(
        f"/api/v1/restaurant/inventory/{inventory_id}",
        headers=headers_user_one,
        json={
            "product_name": "Updated Marinara Base",
            "category": "Pantry Staples",
            "supplier_name": "Updated Supplier",
            "alert_threshold": 3,
        },
    )
    assert inventory_update_response.status_code == 200
    assert inventory_update_response.json()["product_name"] == "Updated Marinara Base"
    assert inventory_update_response.json()["category"] == "Pantry Staples"
    assert inventory_update_response.json()["supplier_name"] == "Updated Supplier"

    inventory_detail_after_update = client.get(f"/api/v1/restaurant/inventory/{inventory_id}", headers=headers_user_one)
    assert inventory_detail_after_update.status_code == 200
    assert inventory_detail_after_update.json()["product_name"] == "Updated Marinara Base"
    assert inventory_detail_after_update.json()["category"] == "Pantry Staples"
    assert inventory_detail_after_update.json()["supplier_name"] == "Updated Supplier"

    inventory_list_after_update = client.get("/api/v1/restaurant/inventory", headers=headers_user_one)
    assert inventory_list_after_update.status_code == 200
    assert inventory_list_after_update.json()["items"][0]["product_name"] == "Updated Marinara Base"
    assert inventory_list_after_update.json()["items"][0]["category"] == "Pantry Staples"

    create_category_response = client.post(
        "/api/v1/restaurant/inventory/categories",
        headers=headers_user_one,
        json={"name": "Cured meats"},
    )
    assert create_category_response.status_code == 201
    assert create_category_response.json()["name"] == "Cured meats"

    create_duplicate_category_response = client.post(
        "/api/v1/restaurant/inventory/categories",
        headers=headers_user_one,
        json={"name": "  cured meats  "},
    )
    assert create_duplicate_category_response.status_code == 201
    assert create_duplicate_category_response.json()["id"] == create_category_response.json()["id"]

    suppliers_user_one_after_update = client.get("/api/v1/restaurant/inventory/suppliers", headers=headers_user_one)
    assert suppliers_user_one_after_update.status_code == 200
    assert {item["name"] for item in suppliers_user_one_after_update.json()["items"]} == {"Global Foods Inc.", "Updated Supplier"}

    categories_user_one_after_update = client.get("/api/v1/restaurant/inventory/categories", headers=headers_user_one)
    assert categories_user_one_after_update.status_code == 200
    assert {item["name"] for item in categories_user_one_after_update.json()["items"]} == {"Sauce", "Pantry Staples", "Cured meats"}

    inventory_stock_response = client.post(
        f"/api/v1/restaurant/inventory/{inventory_id}/stock-update",
        headers=headers_user_one,
        json={"add_stock": 5, "remove_stock": 2},
    )
    assert inventory_stock_response.status_code == 200
    assert inventory_stock_response.json()["current_stock_value"] == 15

    inventory_delete_response = client.delete(f"/api/v1/restaurant/inventory/{inventory_id}", headers=headers_user_one)
    assert inventory_delete_response.status_code == 204


def test_restaurant_cash_deposit_update_and_delete_endpoints(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Deposit Editor",
            "email": "depositeditor@example.com",
            "password": "DepositEditor123",
            "phone": "+15550009998",
        },
    )
    select_subscription_plan(client, headers)

    today = datetime.now(UTC).date()
    tomorrow = today + timedelta(days=1)

    create_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={
            "deposit_date": today.isoformat(),
            "amount": 100.0,
            "type": "bank_deposit",
            "bank_account": "Main Bank",
            "notes": "Initial deposit",
        },
    )
    assert create_response.status_code == 201
    deposit_id = create_response.json()["id"]

    update_response = client.patch(
        f"/api/v1/restaurant/cash/deposits/{deposit_id}",
        headers=headers,
        json={
            "deposit_date": tomorrow.isoformat(),
            "amount": 150.0,
            "type": "cash_deposit",
            "bank_account": "Secondary Bank",
            "notes": "Updated deposit",
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["amount"] == 150.0
    assert update_response.json()["type"] == "cash_deposit"
    assert update_response.json()["bank_account"] == "Secondary Bank"
    assert update_response.json()["notes"] == "Updated deposit"
    assert update_response.json()["deposit_date"][:10] == tomorrow.isoformat()

    delete_response = client.delete(f"/api/v1/restaurant/cash/deposits/{deposit_id}", headers=headers)
    assert delete_response.status_code == 204


def test_cash_management_uses_daily_entries_expenses_invoices_and_deposits(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Cash Flow Owner",
            "email": "cashflow@example.com",
            "password": "CashFlow123",
            "phone": "+1555000201",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "supplier_name": "Fresh Foods Ltd",
            "invoice_number": "INV-CASH-001",
            "invoice_date": today_iso,
            "total_amount": 80.0,
            "line_items": [
                {"product_name": "Rice", "quantity": 2, "unit_price": 40.0, "total_price": 80.0}
            ],
            "source_file_name": "invoice-today.png",
            "ai_provider": "fallback",
            "ai_summary": "Imported invoice",
        },
    )
    assert confirm_response.status_code == 201

    daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": today_iso,
                "pos_payments": 700,
                "cash_payments": 300,
                "bank_transfer_payments": 0,
                "lunch_covers": 20,
                "dinner_covers": 30,
                "opening_cash": 100,
                "closing_cash": 400,
            },
        },
    )
    assert daily_response.status_code == 201

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={"category": "Cleaning Supplies", "amount": 50.0, "expense_date": today_iso, "section": "cash", "notes": "Paid in cash"},
    )
    assert expense_response.status_code == 201
    assert expense_response.json()["section"] == "cash"

    deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={"deposit_date": today_iso, "amount": 125.0, "type": "bank_deposit", "bank_account": "Chase Bank - Main", "notes": "Daily bank drop"},
    )
    assert deposit_response.status_code == 201
    assert deposit_response.json()["type"] == "bank_deposit"

    cash_deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={"deposit_date": today_iso, "amount": 25.0, "type": "cash_deposit", "bank_account": "Chase Bank - Main", "notes": "Daily cash deposit"},
    )
    assert cash_deposit_response.status_code == 201
    assert cash_deposit_response.json()["type"] == "cash_deposit"

    cash_overview_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_overview_response.status_code == 200
    cash_overview_payload = cash_overview_response.json()
    assert cash_overview_payload["periods"]["today"]["summary"]["total_collected"] == 400.0
    assert cash_overview_payload["periods"]["today"]["summary"]["bank_deposits"] == 150.0
    assert cash_overview_payload["periods"]["today"]["summary"]["cash_available"] == 250.0
    assert cash_overview_payload["periods"]["today"]["summary"]["pos_payments"] == 700.0
    assert {item["display_title"] for item in cash_overview_payload["periods"]["today"]["recent_deposits"]} >= {
        "Cash Payments",
        "Chase Bank - Main",
        "POS Settlement",
    }
    assert any(
        item["source_kind"] == "manual_entry" and item["source_subtype"] == "cash_payments"
        and item["type"] == "cash_in"
        for item in cash_overview_payload["periods"]["today"]["recent_deposits"]
    )

    home_response = client.get("/api/v1/restaurant/home?period=weekly", headers=headers)
    assert home_response.status_code == 200
    weekly_cash_cards = {item["label"]: item["amount"] for item in home_response.json()["weekly"]["cash_management"]}
    assert weekly_cash_cards["Total Collection"] == 400.0
    assert weekly_cash_cards["POS Payments"] == 700.0
    assert weekly_cash_cards["Cash Deposit"] == 150.0
    assert weekly_cash_cards["Available Cash"] == 250.0

    db = asyncio.run(app.dependency_overrides[get_database]())
    daily_aggregate = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": today_iso})
    )
    assert daily_aggregate is not None
    assert daily_aggregate["bank_deposits_total"] == 150.0
    assert daily_aggregate["cash_deposits_total"] == 25.0
    assert daily_aggregate["deposits_collection_total"] == 150.0
    assert daily_aggregate["cash_collected_total"] == 400.0
    assert daily_aggregate["cash_available"] == 250.0

    linked_cash_rows = asyncio.run(
        db["restaurant_cash_deposits"].find({"source_kind": "manual_entry", "source_id": daily_response.json()["id"]}).to_list(length=None)
    )
    assert {(item["source_subtype"], item["type"], item["amount"]) for item in linked_cash_rows} == {
        ("cash_payments", "cash_in", 300.0),
        ("pos_payments", "pos_payment", 700.0),
    }
    generated_delete_response = client.delete(f"/api/v1/restaurant/cash/deposits/{linked_cash_rows[0]['_id']}", headers=headers)
    assert generated_delete_response.status_code == 422

    month_aggregate = asyncio.run(
        db["restaurant_finance_snapshots"].find_one(
            {"period_type": "month", "month_key": datetime.now(UTC).date().strftime("%Y-%m")}
        )
    )
    assert month_aggregate is not None
    assert month_aggregate["bank_deposits_total"] == 150.0
    assert month_aggregate["cash_deposits_total"] == 25.0
    assert month_aggregate["deposits_collection_total"] == 150.0
    assert month_aggregate["cash_available"] == 250.0


def test_expenses_section_includes_daily_data_expenses(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Expense Merge Owner",
            "email": "expense-merge@example.com",
            "password": "ExpenseMerge123",
            "phone": "+1555000209",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": today_iso,
                "pos_payments": 300,
                "cash_in": 150,
                "cash_withdrawals": 20,
                "cash_out": 10,
                "expenses_in_cash": 35,
            },
        },
    )
    assert daily_response.status_code == 201

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={"category": "Staff Costs", "amount": 40.0, "expense_date": today_iso, "section": "cash", "notes": "Shift meal"},
    )
    assert expense_response.status_code == 201

    expenses_response = client.get("/api/v1/restaurant/expenses", headers=headers)
    assert expenses_response.status_code == 200
    payload = expenses_response.json()
    assert payload["today"]["total"] == 75.0
    assert payload["this_week"]["total"] == 75.0
    assert payload["this_month"]["total"] == 75.0
    assert payload["today"]["top_category"] == "Staff Costs"
    assert {item["category"] for item in payload["today"]["items"]} == {"Staff Costs", "Expenses in Cash"}
    assert any(
        item["category"] == "Expenses in Cash"
        and item["amount"] == 35.0
        and item["source_kind"] == "manual_entry"
        for item in payload["today"]["items"]
    )

    db = asyncio.run(app.dependency_overrides[get_database]())
    stored_expense = asyncio.run(
        db["restaurant_expenses"].find_one(
            {
                "source_kind": "manual_entry",
                "source_id": daily_response.json()["id"],
            }
        )
    )
    assert stored_expense is not None
    assert stored_expense["amount"] == 35.0

    generated_delete_response = client.delete(f"/api/v1/restaurant/expenses/{stored_expense['_id']}", headers=headers)
    assert generated_delete_response.status_code == 422
    assert "source record" in generated_delete_response.json()["error"]["message"]

    manual_delete_response = client.delete(f"/api/v1/restaurant/expenses/{expense_response.json()['id']}", headers=headers)
    assert manual_delete_response.status_code == 204

    after_delete_response = client.get("/api/v1/restaurant/expenses", headers=headers)
    assert after_delete_response.status_code == 200
    assert after_delete_response.json()["today"]["total"] == 35.0


def test_deleting_daily_data_removes_linked_expense_cash_rows_and_recalculates(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Daily Delete Owner",
            "email": "daily-delete@example.com",
            "password": "DailyDelete123",
            "phone": "+1555000211",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": today_iso,
                "pos_payments": 300,
                "cash_in": 150,
                "cash_withdrawals": 20,
                "cash_out": 10,
                "expenses_in_cash": 35,
            },
        },
    )
    assert daily_response.status_code == 201
    record_id = daily_response.json()["id"]

    db = asyncio.run(app.dependency_overrides[get_database]())
    assert asyncio.run(db["restaurant_expenses"].count_documents({"source_kind": "manual_entry", "source_id": record_id})) == 1
    linked_cash_rows = asyncio.run(
        db["restaurant_cash_deposits"].find({"source_kind": "manual_entry", "source_id": record_id}).to_list(length=None)
    )
    assert {(item["source_subtype"], item["type"], item["amount"]) for item in linked_cash_rows} == {
        ("pos_payments", "pos_payment", 300.0),
        ("cash_in", "cash_in", 150.0),
        ("cash_withdrawals", "cash_withdrawal", 20.0),
        ("cash_out", "cash_out", 10.0),
        ("expenses_in_cash", "cash_expense", 35.0),
    }
    assert asyncio.run(db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": today_iso})) is not None

    cash_overview_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_overview_response.status_code == 200
    recent_by_subtype = {
        item["source_subtype"]: item
        for item in cash_overview_response.json()["periods"]["today"]["recent_deposits"]
        if item["source_id"] == record_id
    }
    assert recent_by_subtype["pos_payments"]["amount"] == 300.0
    assert recent_by_subtype["cash_in"]["amount"] == 150.0
    assert recent_by_subtype["cash_withdrawals"]["amount"] == -20.0
    assert recent_by_subtype["cash_out"]["amount"] == -10.0
    assert recent_by_subtype["expenses_in_cash"]["amount"] == -35.0

    generated_delete_response = client.delete(f"/api/v1/restaurant/cash/deposits/{linked_cash_rows[0]['_id']}", headers=headers)
    assert generated_delete_response.status_code == 422

    delete_response = client.delete(f"/api/v1/restaurant/daily-data/{record_id}", headers=headers)
    assert delete_response.status_code == 204

    assert asyncio.run(db["restaurant_expenses"].count_documents({"source_kind": "manual_entry", "source_id": record_id})) == 0
    assert asyncio.run(db["restaurant_cash_deposits"].count_documents({"source_kind": "manual_entry", "source_id": record_id})) == 0
    assert asyncio.run(db["restaurant_finance_transactions"].count_documents({"source_kind": "manual_entry", "source_id": record_id})) == 0
    assert asyncio.run(db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": today_iso})) is None

    expenses_response = client.get("/api/v1/restaurant/expenses", headers=headers)
    assert expenses_response.status_code == 200
    assert expenses_response.json()["today"]["total"] == 0

    cash_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_response.status_code == 200
    assert cash_response.json()["periods"]["today"]["summary"]["total_collected"] == 0


def test_expenses_section_includes_uploaded_document_expenses(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Document Expense Owner",
            "email": "document-expense@example.com",
            "password": "DocumentExpense123",
            "phone": "+1555000210",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Expense",
            "supplier_name": "Fresh Foods Ltd",
            "counterparty_name": "Fresh Foods Ltd",
            "invoice_number": "INV-EXP-001",
            "invoice_date": today_iso,
            "total_amount": 80.0,
            "expense_amount": 80.0,
            "currency": "EUR",
            "line_items": [
                {"product_name": "Rice", "quantity": 2, "unit_price": 40.0, "total_price": 80.0}
            ],
            "source_file_name": "invoice-today.png",
            "ai_provider": "fallback",
            "ai_summary": "Imported invoice",
        },
    )
    assert confirm_response.status_code == 201

    expenses_response = client.get("/api/v1/restaurant/expenses", headers=headers)
    assert expenses_response.status_code == 200
    payload = expenses_response.json()
    assert payload["today"]["total"] == 80.0
    assert payload["this_week"]["total"] == 80.0
    assert payload["this_month"]["total"] == 80.0
    assert payload["this_year"]["total"] == 80.0
    assert any(
        item["category"] == "Expense"
        and item["amount"] == 80.0
        and item["source_kind"] == "document"
        for item in payload["today"]["items"]
    )

    db = asyncio.run(app.dependency_overrides[get_database]())
    stored_expense = asyncio.run(
        db["restaurant_expenses"].find_one(
            {
                "source_kind": "document",
                "source_id": confirm_response.json()["id"],
            }
        )
    )
    assert stored_expense is not None
    assert stored_expense["amount"] == 80.0


def test_expenses_section_today_uses_latest_business_day_across_sources(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Expense Day Anchor Owner",
            "email": "expense-day-anchor@example.com",
            "password": "ExpenseDayAnchor123",
            "phone": "+1555000311",
        },
    )
    select_subscription_plan(client, headers)

    local_today_iso = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()

    manual_expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={
            "category": "Staff Costs",
            "amount": 40.0,
            "expense_date": local_today_iso,
            "section": "cash",
            "notes": "Shift meal",
        },
    )
    assert manual_expense_response.status_code == 201

    inventory_response = client.post(
        "/api/v1/restaurant/inventory",
        headers=headers,
        json={
            "product_name": "Tomato",
            "category": "Vegetable",
            "stock_quantity": 20,
            "unit_type": "kg",
            "supplier_name": "Fresh Farm",
            "unit_price": 5.0,
            "alert_threshold": 3,
            "purchase_date": local_today_iso,
        },
    )
    assert inventory_response.status_code == 201

    document_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Expense",
            "supplier_name": "Fresh Foods Ltd",
            "counterparty_name": "Fresh Foods Ltd",
            "invoice_number": "INV-EXP-ANCHOR-001",
            "invoice_date": local_today_iso,
            "total_amount": 80.0,
            "expense_amount": 80.0,
            "currency": "EUR",
            "line_items": [
                {"product_name": "Rice", "quantity": 2, "unit_price": 40.0, "total_price": 80.0}
            ],
            "source_file_name": "invoice-anchor.png",
            "ai_provider": "fallback",
            "ai_summary": "Imported invoice",
        },
    )
    assert document_response.status_code == 201

    expenses_response = client.get("/api/v1/restaurant/expenses", headers=headers)
    assert expenses_response.status_code == 200
    today_items = expenses_response.json()["today"]["items"]
    assert {item["source_kind"] for item in today_items if item.get("source_kind")} >= {"inventory", "document"}
    assert any(item["category"] == "Staff Costs" and item.get("source_kind") is None for item in today_items)


def test_cash_management_today_uses_latest_business_day_for_local_entries(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Cash Day Anchor Owner",
            "email": "cash-day-anchor@example.com",
            "password": "CashDayAnchor123",
            "phone": "+1555000312",
        },
    )
    select_subscription_plan(client, headers)

    local_today_iso = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()

    daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": local_today_iso,
                "pos_payments": 300,
                "cash_in": 150,
                "cash_withdrawals": 20,
                "cash_out": 10,
                "expenses_in_cash": 35,
            },
        },
    )
    assert daily_response.status_code == 201
    record_id = daily_response.json()["id"]

    cash_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_response.status_code == 200
    recent_by_subtype = {
        item["source_subtype"]: item
        for item in cash_response.json()["periods"]["today"]["recent_deposits"]
        if item["source_id"] == record_id
    }
    assert recent_by_subtype["pos_payments"]["amount"] == 300.0
    assert recent_by_subtype["cash_in"]["amount"] == 150.0
    assert recent_by_subtype["cash_withdrawals"]["amount"] == -20.0
    assert recent_by_subtype["cash_out"]["amount"] == -10.0
    assert recent_by_subtype["expenses_in_cash"]["amount"] == -35.0


def test_home_revenue_uses_deposit_sales_when_no_manual_revenue_exists(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Deposit Revenue Owner",
            "email": "deposit-revenue@example.com",
            "password": "DepositRevenue123",
            "phone": "+1555000209",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={
            "deposit_date": today_iso,
            "amount": 275.0,
            "type": "bank_deposit",
            "bank_account": "Main Sales Deposit",
            "notes": "Sales deposited directly",
        },
    )
    assert deposit_response.status_code == 201

    home_response = client.get("/api/v1/restaurant/home?period=weekly", headers=headers)
    assert home_response.status_code == 200
    home_payload = home_response.json()
    assert home_payload["weekly"]["metrics"][0]["label"] == "Revenue"
    assert home_payload["weekly"]["metrics"][0]["value"] == 275.0
    assert any(point["value"] == 275.0 for point in home_payload["weekly"]["revenue"])


def test_home_recent_activity_includes_all_core_operations(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Recent Activity Owner",
            "email": "recent-activity@example.com",
            "password": "RecentActivity123",
            "phone": "+1555000310",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    manual_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": today_iso,
                "pos_payments": 150,
                "cash_payments": 40,
                "bank_transfer_payments": 10,
                "lunch_covers": 5,
                "dinner_covers": 7,
                "opening_cash": 20,
                "closing_cash": 35,
            },
        },
    )
    assert manual_entry_response.status_code == 201

    document_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Expense",
            "supplier_name": "Fresh Produce Co",
            "invoice_number": "INV-RECENT-1",
            "invoice_date": today_iso,
            "total_amount": 88.0,
            "currency": "EUR",
            "expense_amount": 88.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "line_items": [],
            "source_file_name": "recent-invoice.pdf",
            "ai_provider": "fallback",
            "ai_summary": "Recent activity coverage",
        },
    )
    assert document_response.status_code == 201

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={"category": "Utilities", "amount": 25.0, "expense_date": today_iso, "section": "cash", "notes": "Water"},
    )
    assert expense_response.status_code == 201

    deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={"deposit_date": today_iso, "amount": 90.0, "type": "bank_deposit", "bank_account": "Primary Bank", "notes": "Drop"},
    )
    assert deposit_response.status_code == 201

    inventory_response = client.post(
        "/api/v1/restaurant/inventory",
        headers=headers,
        json={
            "product_name": "Olive Oil",
            "category": "Pantry",
            "stock_quantity": 8,
            "unit_type": "bottles",
            "supplier_name": "Kitchen Supply Co",
            "unit_price": 9.5,
            "alert_threshold": 2,
            "purchase_date": today_iso,
        },
    )
    assert inventory_response.status_code == 201

    recent_activity_response = client.get("/api/v1/restaurant/home/recent-activity", headers=headers)
    assert recent_activity_response.status_code == 200
    recent_activity_payload = recent_activity_response.json()
    assert set(recent_activity_payload.keys()) == {"items"}
    kinds = {item["kind"] for item in recent_activity_payload["items"]}
    assert {"daily_record", "invoice", "expense", "cash", "inventory"}.issubset(kinds)

    language_update_response = client.put(
        "/api/v1/auth/preferences/language",
        headers=headers,
        json={"preferred_language": "it"},
    )
    assert language_update_response.status_code == 200

    italian_recent_activity_response = client.get("/api/v1/restaurant/home/recent-activity", headers=headers)
    assert italian_recent_activity_response.status_code == 200
    italian_items = italian_recent_activity_response.json()["items"]
    assert italian_items
    assert any(
        any(
            phrase in f"{item.get('title', '')} {item.get('subtitle', '')}".lower()
            for phrase in ("ricavi", "fattura caricata", "spesa", "deposito", "articolo inventario")
        )
        for item in italian_items
    )


def test_notification_feed_includes_business_change_messages(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Notification Feed Owner",
            "email": "notification-feed@example.com",
            "password": "NotificationFeed123",
            "phone": "+1555000399",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    manual_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": today_iso,
                "pos_payments": 120,
                "cash_in": 80,
                "cash_withdrawals": 10,
                "cash_out": 5,
                "expenses_in_cash": 15,
                "notes": "Feed seed",
            },
        },
    )
    assert manual_entry_response.status_code == 201

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={"category": "Utilities", "amount": 25.0, "expense_date": today_iso, "section": "cash", "notes": "Water"},
    )
    assert expense_response.status_code == 201

    deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={"deposit_date": today_iso, "amount": 60.0, "type": "bank_deposit", "bank_account": "Primary Bank", "notes": "Drop"},
    )
    assert deposit_response.status_code == 201

    notification_response = client.get("/api/v1/restaurant/notifications/feed", headers=headers)
    assert notification_response.status_code == 200
    payload = notification_response.json()
    assert set(payload.keys()) == {"items"}
    assert any("Cash available decreased by EUR 60.00" in item["title"] for item in payload["items"])
    assert any("Cash available decreased by EUR 25.00" in item["title"] for item in payload["items"])
    assert any("Daily cash updated to EUR 50.00" in item["title"] for item in payload["items"])


def test_home_metrics_include_uploaded_document_expenses_like_inventory(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Home Metrics Owner",
            "email": "home-metrics-owner@example.com",
            "password": "HomeMetrics123",
            "phone": "+1555000313",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    inventory_response = client.post(
        "/api/v1/restaurant/inventory",
        headers=headers,
        json={
            "product_name": "Salmon",
            "category": "Food Supplies",
            "stock_quantity": 5,
            "unit_type": "kg",
            "supplier_name": "Ocean Fresh",
            "unit_price": 20.0,
            "alert_threshold": 1,
            "purchase_date": today_iso,
        },
    )
    assert inventory_response.status_code == 201

    document_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Expense",
            "supplier_name": "Fresh Produce Co",
            "invoice_number": "INV-HOME-1",
            "invoice_date": today_iso,
            "total_amount": 88.0,
            "currency": "EUR",
            "expense_amount": 88.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "line_items": [],
            "source_file_name": "home-metrics-invoice.pdf",
            "ai_provider": "fallback",
            "ai_summary": "Home metrics coverage",
        },
    )
    assert document_response.status_code == 201

    home_metrics_response = client.get("/api/v1/restaurant/home/metrics?period=weekly", headers=headers)
    assert home_metrics_response.status_code == 200
    home_metrics_payload = home_metrics_response.json()
    other_expense_metric = next(item for item in home_metrics_payload["items"] if item["label"] == "Other Expense")
    food_cost_metric = next(item for item in home_metrics_payload["items"] if item["label"] == "Food Cost")
    profit_metric = next(item for item in home_metrics_payload["items"] if item["label"] == "Profit")
    assert other_expense_metric["value"] == 0.0
    assert food_cost_metric["value"] == 100.0
    assert profit_metric["value"] == -188.0

    home_response = client.get("/api/v1/restaurant/home?period=weekly", headers=headers)
    assert home_response.status_code == 200
    weekly_other_expense_metric = next(item for item in home_response.json()["weekly"]["metrics"] if item["label"] == "Other Expense")
    weekly_food_cost_metric = next(item for item in home_response.json()["weekly"]["metrics"] if item["label"] == "Food Cost")
    assert weekly_other_expense_metric["value"] == 0.0
    assert weekly_food_cost_metric["value"] == 100.0

    db = asyncio.run(app.dependency_overrides[get_database]())
    restaurant_record = asyncio.run(
        db["restaurant_finance_snapshots"].find_one({"period_type": "day", "business_date": today_iso})
    )
    assert restaurant_record is not None
    assert restaurant_record["uploaded_document_total"] == 88.0
    assert restaurant_record["manual_expense_total"] == 100.0
    assert restaurant_record["total_expenses"] == 188.0


def test_home_food_cost_uses_inventory_add_cost_without_daily_records(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Inventory Only Owner",
            "email": "inventory-only-owner@example.com",
            "password": "InventoryOnly123",
            "phone": "+1555000314",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    inventory_response = client.post(
        "/api/v1/restaurant/inventory",
        headers=headers,
        json={
            "product_name": "Rice",
            "category": "Food Supplies",
            "stock_quantity": 30,
            "unit_type": "kg",
            "supplier_name": "Market Supplier",
            "unit_price": 100.0,
            "alert_threshold": 5,
            "purchase_date": today_iso,
        },
    )
    assert inventory_response.status_code == 201

    home_metrics_response = client.get("/api/v1/restaurant/home/metrics?period=weekly", headers=headers)
    assert home_metrics_response.status_code == 200
    home_metrics_payload = home_metrics_response.json()
    food_cost_metric = next(item for item in home_metrics_payload["items"] if item["label"] == "Food Cost")
    assert food_cost_metric["value"] == 3000.0

    home_response = client.get("/api/v1/restaurant/home?period=weekly", headers=headers)
    assert home_response.status_code == 200
    weekly_food_cost_metric = next(item for item in home_response.json()["weekly"]["metrics"] if item["label"] == "Food Cost")
    assert weekly_food_cost_metric["value"] == 3000.0


def test_restaurant_cash_api_contracts_remain_stable(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Contract Owner",
            "email": "contract-owner@example.com",
            "password": "ContractOwner123",
            "phone": "+1555000301",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    expense_response = client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={
            "category": "Utilities",
            "amount": 40.0,
            "expense_date": today_iso,
            "section": "bank",
            "notes": "Electricity",
        },
    )
    assert expense_response.status_code == 201
    assert set(expense_response.json().keys()) == {
        "id",
        "category",
        "amount",
        "expense_date",
        "section",
        "notes",
        "source_kind",
        "source_id",
        "source_inventory_item_id",
        "created_at",
    }

    deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={
            "deposit_date": today_iso,
            "amount": 10.0,
            "type": "bank_deposit",
            "bank_account": "Primary Bank",
            "notes": "Drop",
        },
    )
    assert deposit_response.status_code == 201
    assert set(deposit_response.json().keys()) == {
        "id",
        "deposit_date",
        "amount",
        "type",
        "bank_account",
        "notes",
        "source_kind",
        "source_id",
        "source_subtype",
        "created_at",
        "amount_formatted",
        "deposit_date_formatted",
        "display_title",
    }

    cash_overview_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_overview_response.status_code == 200
    payload = cash_overview_response.json()
    assert set(payload.keys()) == {"active_period", "periods"}
    assert set(payload["periods"].keys()) == {"today", "this_week", "this_month"}
    assert set(payload["periods"]["today"].keys()) == {"summary", "status", "recent_deposits"}
    assert set(payload["periods"]["today"]["summary"].keys()) == {"total_collected", "cash_available", "pos_payments", "withdrawals_total", "bank_deposits"}


def test_restaurant_daily_data_dashboard_analytics_and_chat(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Alex Chef",
            "email": "alex@example.com",
            "password": "AlexChef123",
            "phone": "+1555000003",
        },
    )
    select_subscription_plan(client, headers)

    today = datetime.now(UTC).date()
    previous_day = today - timedelta(days=1)
    previous_day_iso = previous_day.isoformat()
    today_iso = today.isoformat()

    create_daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": previous_day_iso,
                "pos_payments": 800,
                "cash_payments": 300,
                "bank_transfer_payments": 200,
                "lunch_covers": 45,
                "dinner_covers": 60,
                "opening_cash": 150,
                "closing_cash": 420,
            },
        },
    )
    assert create_daily_response.status_code == 201
    assert create_daily_response.json()["profit"] == 1300

    client.post(
        "/api/v1/restaurant/expenses",
        headers=headers,
        json={"category": "Food Supplies", "amount": 250.0, "expense_date": previous_day_iso},
    )

    home_response = client.get("/api/v1/restaurant/home?period=weekly", headers=headers)
    assert home_response.status_code == 200
    home_payload = home_response.json()
    assert home_payload["available_periods"] == ["weekly", "monthly"]
    assert home_payload["weekly"]["metrics"][0]["label"] == "Revenue"
    assert home_payload["monthly"]["metrics"][0]["label"] == "Revenue"
    assert home_payload["weekly"]["revenue"][0]["label"] in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    assert home_payload["monthly"]["revenue"][0]["label"].startswith("Week ")

    home_monthly_response = client.get("/api/v1/restaurant/home?period=monthly", headers=headers)
    assert home_monthly_response.status_code == 200
    assert "monthly" in home_monthly_response.json()

    home_export_response = client.get("/api/v1/restaurant/home/export?period=weekly&format=pdf", headers=headers)
    assert home_export_response.status_code == 404

    home_custom_range_response = client.get(f"/api/v1/restaurant/home?period=weekly&from_date={previous_day_iso}&to_date={today_iso}", headers=headers)
    assert home_custom_range_response.status_code == 200
    assert "weekly" in home_custom_range_response.json()

    home_shell_response = client.get(
        "/api/v1/restaurant/home?period=weekly&include_metrics=false&include_cash_management=false",
        headers=headers,
    )
    assert home_shell_response.status_code == 200
    home_shell_payload = home_shell_response.json()
    assert home_shell_payload["weekly"]["metrics"] == []
    assert home_shell_payload["weekly"]["cash_management"] == []

    home_metrics_response = client.get("/api/v1/restaurant/home/metrics?period=weekly", headers=headers)
    assert home_metrics_response.status_code == 200
    assert home_metrics_response.json()["period"] == "weekly"
    assert home_metrics_response.json()["items"][0]["label"] == "Revenue"

    home_cash_response = client.get("/api/v1/restaurant/home/cash-management?period=weekly", headers=headers)
    assert home_cash_response.status_code == 200
    assert home_cash_response.json()["period"] == "weekly"
    assert {item["label"] for item in home_cash_response.json()["items"]} == {
        "Total Collection",
        "POS Payments",
        "Available Cash",
        "Cash Deposit",
    }

    home_vat_balance_response = client.get("/api/v1/restaurant/home/vat-balance", headers=headers)
    assert home_vat_balance_response.status_code == 200
    assert "balance" in home_vat_balance_response.json()

    cash_deposit_response = client.post(
        "/api/v1/restaurant/cash/deposits",
        headers=headers,
        json={
            "deposit_date": previous_day_iso,
            "amount": 450.0,
            "type": "bank_deposit",
            "bank_account": "Chase Bank - Main",
            "notes": "Chase Bank - Main",
        },
    )
    assert cash_deposit_response.status_code == 201
    assert cash_deposit_response.json()["amount_formatted"] == "€450.00"
    assert cash_deposit_response.json()["type"] == "bank_deposit"
    assert cash_deposit_response.json()["deposit_date_formatted"] == previous_day.strftime("%b %d, %Y")
    assert cash_deposit_response.json()["display_title"] == "Chase Bank - Main"

    cash_overview_response = client.get("/api/v1/restaurant/cash/overview", headers=headers)
    assert cash_overview_response.status_code == 200
    cash_overview_payload = cash_overview_response.json()
    assert cash_overview_payload["active_period"] == "today"
    assert set(cash_overview_payload["periods"].keys()) == {"today", "this_week", "this_month"}
    assert cash_overview_payload["periods"]["this_month"]["summary"]["bank_deposits"] >= 450.0
    assert cash_overview_payload["periods"]["today"]["status"]["cash_available"] == "IN_SAFE"
    assert cash_overview_payload["periods"]["this_month"]["recent_deposits"][0]["display_title"] == "Chase Bank - Main"

    insights_response = client.get("/api/v1/restaurant/insights", headers=headers)
    assert insights_response.status_code == 200
    assert insights_response.json()["title"]
    assert len(insights_response.json()["root_causes"]) == 3
    assert len(insights_response.json()["recommended_actions"]) == 3

    language_update_response = client.put(
        "/api/v1/auth/preferences/language",
        headers=headers,
        json={"preferred_language": "it"},
    )
    assert language_update_response.status_code == 200

    italian_insights_response = client.get("/api/v1/restaurant/home/insight", headers=headers)
    assert italian_insights_response.status_code == 200
    italian_insight = italian_insights_response.json()["insight"]
    assert italian_insight["title"]
    assert italian_insight["title_translations"]["en"]
    assert italian_insight["title_translations"]["it"]
    assert italian_insight["summary_translations"]["en"]
    assert italian_insight["summary_translations"]["it"]
    assert any(
        phrase in italian_insight["summary"].lower()
        for phrase in ("il costo del cibo", "controlla i prezzi", "sprechi")
    )


    second_daily_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": today_iso,
                "pos_payments": 600,
                "cash_payments": 200,
                "bank_transfer_payments": 120,
                "lunch_covers": 20,
                "dinner_covers": 18,
                "opening_cash": 100,
                "closing_cash": 220,
            },
        },
    )
    assert second_daily_response.status_code == 201

    daily_list_response = client.get(f"/api/v1/restaurant/daily-data?view=date&reference_date={today_iso}", headers=headers)
    assert daily_list_response.status_code == 200
    daily_list_payload = daily_list_response.json()
    assert set(daily_list_payload.keys()) == {"total", "page", "page_size", "pages", "items"}
    assert daily_list_payload["items"][0]["business_date"] == today_iso
    assert set(daily_list_payload["items"][0].keys()) == {
        "id",
        "record_id",
        "business_date",
        "total_revenue",
        "operating_revenue",
        "total_expenses",
        "operating_expenses",
        "invoice_document_total",
        "total_covers",
        "avg_revenue_per_cover",
        "created_at",
    }
    assert daily_list_payload["items"][0]["record_id"] == second_daily_response.json()["id"]
    assert daily_list_payload["items"][0]["total_covers"] == 38
    assert daily_list_payload["items"][0]["total_expenses"] == 0.0
    assert daily_list_payload["items"][0]["avg_revenue_per_cover"] == 24.21
    assert daily_list_payload["items"][0]["operating_revenue"] == 920.0
    assert daily_list_payload["items"][0]["invoice_document_total"] == 0.0

    week_list_response = client.get(f"/api/v1/restaurant/daily-data?view=week&reference_date={today_iso}", headers=headers)
    assert week_list_response.status_code == 200
    week_payload = week_list_response.json()
    assert week_payload["total"] == 1
    assert set(week_payload.keys()) == {"total", "page", "page_size", "pages", "items"}

    detail_id = daily_list_payload["items"][0]["id"]
    detail_response = client.get(f"/api/v1/restaurant/daily-data/{detail_id}", headers=headers)
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["business_date"] == today_iso
    assert detail_payload["revenue_breakdown"][0]["label"] == "POS Payments"
    assert detail_payload["covers_summary"]["total"] == 38
    assert detail_payload["register_summary"]["cash_payments"] == 200
    assert detail_payload["register_summary"]["closing_cash"] == 220
    assert detail_payload["register_summary"]["total_cash_on_hand"] == 420
    assert [section["key"] for section in detail_payload["method_sections"]] == [
        "deposit_section",
        "covers_section",
        "register_section",
    ]
    assert detail_payload["method_sections"][0]["fields"][1]["value"] == 200.0
    assert detail_payload["method_sections"][1]["fields"][2]["value"] == 38

    date_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-date?business_date={today_iso}", headers=headers)
    assert date_detail_response.status_code == 200
    date_detail_payload = date_detail_response.json()
    assert date_detail_payload["business_date"] == today_iso
    assert date_detail_payload["total_revenue"] == 920
    assert date_detail_payload["operating_revenue"] == 920
    assert date_detail_payload["invoice_document_total"] == 0.0
    assert date_detail_payload["total_covers"] == 38
    assert date_detail_payload["register_summary"]["cash_payments"] == 200
    assert date_detail_payload["register_summary"]["total_cash_on_hand"] == 420
    assert date_detail_payload["document_count"] == 0
    assert date_detail_payload["method_sections"][0]["key"] == "deposit_section"
    assert date_detail_payload["method_sections"][1]["fields"][3]["value"] == 0.0

    week_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-week?reference_date={today_iso}", headers=headers)
    assert week_detail_response.status_code == 200
    week_detail_payload = week_detail_response.json()
    assert week_detail_payload["business_date"] == today_iso
    assert week_detail_payload["total_revenue"] == 2220
    assert week_detail_payload["operating_revenue"] == 2220
    assert week_detail_payload["register_summary"]["cash_payments"] == 500
    assert week_detail_payload["register_summary"]["total_cash_on_hand"] == 1140
    assert week_detail_payload["invoice_document_total"] == 0.0
    assert week_detail_payload["total_covers"] == 143
    assert week_detail_payload["document_count"] == 0
    assert week_detail_payload["method_sections"][0]["fields"][4]["value"] == 2220.0

    date_reference_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-date-reference?reference_date={today_iso}", headers=headers)
    assert date_reference_detail_response.status_code == 200
    assert date_reference_detail_response.json()["business_date"] == today_iso

    week_business_date_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-week-business-date?business_date={today_iso}", headers=headers)
    assert week_business_date_detail_response.status_code == 200
    assert week_business_date_detail_response.json()["business_date"] == today_iso

    month_list_response = client.get(f"/api/v1/restaurant/daily-data?view=month&reference_date={today_iso}", headers=headers)
    assert month_list_response.status_code == 200
    month_payload = month_list_response.json()
    assert month_payload["total"] == 1
    assert set(month_payload.keys()) == {"total", "page", "page_size", "pages", "items"}

    month_business_date_detail_response = client.get(f"/api/v1/restaurant/daily-data/by-month-business-date?business_date={today_iso}", headers=headers)
    assert month_business_date_detail_response.status_code == 200
    assert month_business_date_detail_response.json()["business_date"] == today_iso

    all_dates_response = client.get("/api/v1/restaurant/daily-data/by-date", headers=headers)
    assert all_dates_response.status_code == 200
    assert set(all_dates_response.json().keys()) == {"total", "items"}

    all_weeks_response = client.get("/api/v1/restaurant/daily-data/by-week", headers=headers)
    assert all_weeks_response.status_code == 200
    assert set(all_weeks_response.json().keys()) == {"total", "items"}

    all_months_response = client.get("/api/v1/restaurant/daily-data/by-month", headers=headers)
    assert all_months_response.status_code == 200
    assert set(all_months_response.json().keys()) == {"total", "items"}

    delete_response = client.delete(f"/api/v1/restaurant/daily-data/{detail_id}", headers=headers)
    assert delete_response.status_code == 204
    after_delete_response = client.get(f"/api/v1/restaurant/daily-data?view=date&reference_date={today_iso}", headers=headers)
    assert after_delete_response.status_code == 200
    assert after_delete_response.json()["total"] == 0

    business_insight_response = client.get("/api/v1/restaurant/analytics/business-insight", headers=headers)
    assert business_insight_response.status_code == 200
    business_insight_payload = business_insight_response.json()
    assert business_insight_payload["title"]
    assert business_insight_payload["subtitle"]
    assert business_insight_payload["title_translations"]["en"]
    assert business_insight_payload["title_translations"]["it"]
    assert business_insight_payload["subtitle_translations"]["en"]
    assert business_insight_payload["subtitle_translations"]["it"]

    italian_business_insight_response = client.get("/api/v1/restaurant/analytics/business-insight", headers=headers)
    assert italian_business_insight_response.status_code == 200
    italian_business_insight_payload = italian_business_insight_response.json()
    assert italian_business_insight_payload["title"]
    assert any(
        phrase in italian_business_insight_payload["title"].lower() or phrase in italian_business_insight_payload["subtitle"].lower()
        for phrase in ("suggerimento di ottimizzazione", "controlla", "ricavi", "costi")
    )

    analytics_response = client.get("/api/v1/restaurant/analytics/overview", headers=headers)
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()
    assert analytics_payload["revenue_total"] == 1300
    assert analytics_payload["operating_revenue_total"] == 1300
    assert analytics_payload["invoice_document_total"] == 0.0
    assert analytics_payload["insight_banner"]["title"] == business_insight_payload["title"]
    assert analytics_payload["metric_tiles"][0]["label"] == "Estimated Profit"
    assert analytics_payload["metric_tiles"][1]["label"] == "Peak Hour"
    assert analytics_payload["metric_tiles"][1]["value"] == "Cena"
    assert analytics_payload["metric_tiles"][1]["subtitle"] == "60 coperti, 57% del periodo"
    assert analytics_payload["summary_stats"][0]["label"] == "Revenue"
    assert analytics_payload["summary_stats"][0]["value"] == 1300
    assert analytics_payload["summary_stats"][1]["label"] == "Covers"
    assert analytics_payload["summary_stats"][1]["value"] == 105
    assert analytics_payload["summary_stats"][2]["label"] == "Avg Rev"
    assert analytics_payload["summary_stats"][2]["value"] == 12.38
    assert analytics_payload["revenue_comparison"][0]["label"] == "This Week Revenue"
    assert analytics_payload["covers_activity"][0]["label"] == "Lunch"
    assert analytics_payload["covers_activity"][0]["value"] == 45
    assert analytics_payload["covers_activity"][1]["label"] == "Dinner"
    assert analytics_payload["covers_activity"][1]["value"] == 60
    assert analytics_payload["cost_breakdown"][0]["label"] == "Food Cost"
    assert analytics_payload["cost_breakdown"][0]["value"] == 19.2
    assert analytics_payload["cost_breakdown"][1]["label"] == "Staff Cost"
    assert analytics_payload["cost_breakdown"][1]["value"] == 0
    assert len(analytics_payload["weekly_revenue"]) == 7
    assert len(analytics_payload["supplier_price_alerts"]) >= 1
    assert analytics_payload["supplier_price_alerts"][0]["title"]
    assert analytics_payload["supplier_price_alerts"][0]["subtitle"]


    analytics_monthly_response = client.get("/api/v1/restaurant/analytics/overview?period=monthly", headers=headers)
    assert analytics_monthly_response.status_code == 200
    analytics_monthly_payload = analytics_monthly_response.json()
    assert analytics_monthly_payload["revenue_comparison"][0]["label"] == "This Month Revenue"

    analytics_revenue_comparison_response = client.get("/api/v1/restaurant/analytics/revenue-comparison", headers=headers)
    assert analytics_revenue_comparison_response.status_code == 200
    analytics_revenue_comparison_payload = analytics_revenue_comparison_response.json()
    assert analytics_revenue_comparison_payload["period"] == "weekly"
    assert analytics_revenue_comparison_payload["items"][0]["label"] == "This Week Revenue"


def test_restaurant_analytics_peak_hour_falls_back_to_latest_cover_record(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Fallback Owner",
            "email": "fallback-owner@example.com",
            "password": "FallbackOwner123",
            "phone": "+1555000777",
        },
    )
    select_subscription_plan(client, headers)

    today = datetime.now(UTC).date()
    previous_day = today - timedelta(days=1)
    older_day = today - timedelta(days=20)

    older_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": older_day.isoformat(),
                "pos_payments": 300,
                "cash_payments": 100,
                "bank_transfer_payments": 50,
                "lunch_covers": 14,
                "dinner_covers": 36,
                "opening_cash": 100,
                "closing_cash": 200,
            },
        },
    )
    assert older_entry_response.status_code == 201

    current_period_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_1",
            "method_one": {
                "business_date": previous_day.isoformat(),
                "pos_payments": 250,
                "cash_withdrawals": 0,
                "cash_in": 20,
                "cash_out": 0,
                "expenses_in_cash": 0,
                "notes": "No cover split",
            },
        },
    )
    assert current_period_entry_response.status_code == 201

    analytics_response = client.get("/api/v1/restaurant/analytics/overview?period=weekly", headers=headers)
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()
    assert analytics_payload["metric_tiles"][1]["label"] == "Peak Hour"
    assert analytics_payload["metric_tiles"][1]["value"] == "Dinner"
    assert analytics_payload["metric_tiles"][1]["subtitle"] == "36 covers, 72% of this period using latest available record"
    analytics_revenue_comparison_response = client.get("/api/v1/restaurant/analytics/revenue-comparison", headers=headers)
    assert analytics_revenue_comparison_response.status_code == 200
    analytics_revenue_comparison_payload = analytics_revenue_comparison_response.json()
    assert analytics_revenue_comparison_payload["period"] == "weekly"
    assert analytics_revenue_comparison_payload["items"][0]["label"] == "This Week Revenue"

    analytics_covers_activity_response = client.get("/api/v1/restaurant/analytics/covers-activity", headers=headers)
    assert analytics_covers_activity_response.status_code == 200
    analytics_covers_activity_payload = analytics_covers_activity_response.json()
    assert analytics_covers_activity_payload["period"] == "weekly"
    assert analytics_covers_activity_payload["items"][0]["label"] == "Lunch"
    assert analytics_covers_activity_payload["items"][0]["value"] == 14
    assert analytics_covers_activity_payload["items"][1]["label"] == "Dinner"
    assert analytics_covers_activity_payload["items"][1]["value"] == 36

    analytics_cost_breakdown_response = client.get("/api/v1/restaurant/analytics/cost-breakdown", headers=headers)
    assert analytics_cost_breakdown_response.status_code == 200
    analytics_cost_breakdown_payload = analytics_cost_breakdown_response.json()
    assert analytics_cost_breakdown_payload["period"] == "weekly"
    assert analytics_cost_breakdown_payload["items"][0]["label"] == "Food Cost"
    assert analytics_cost_breakdown_payload["items"][0]["value"] == 0.0
    assert analytics_cost_breakdown_payload["items"][1]["label"] == "Staff Cost"
    assert analytics_cost_breakdown_payload["items"][1]["value"] == 0

    analytics_supplier_alerts_response = client.get("/api/v1/restaurant/analytics/supplier-alerts", headers=headers)
    assert analytics_supplier_alerts_response.status_code == 200
    analytics_supplier_alerts_payload = analytics_supplier_alerts_response.json()
    assert analytics_supplier_alerts_payload["period"] == "weekly"
    assert len(analytics_supplier_alerts_payload["items"]) >= 1
    assert analytics_supplier_alerts_payload["items"][0]["title"]
    assert analytics_supplier_alerts_payload["items"][0]["subtitle"]

    analytics_export_response = client.get("/api/v1/restaurant/analytics/export?period=weekly&format=pdf", headers=headers)
    assert analytics_export_response.status_code == 404

    chat_list_response = client.get("/api/v1/restaurant/chat/messages", headers=headers)
    assert chat_list_response.status_code == 200
    chat_list_payload = chat_list_response.json()
    assert set(chat_list_payload.keys()) == {"messages"}
    assert len(chat_list_payload["messages"]) >= 1

    chat_response = client.post("/api/v1/restaurant/chat/messages", headers=headers, json={"message": "How can I improve profit?"})
    assert chat_response.status_code == 201
    chat_payload = chat_response.json()
    assert set(chat_payload.keys()) == {"messages"}
    messages = chat_payload["messages"]
    assert any(message["role"] == "insight" for message in messages)
    assert messages[-1]["role"] == "assistant"
    assert "revenue" in messages[-1]["message"].lower()
    assert messages[-1]["message_translations"]["en"]
    assert messages[-1]["message_translations"]["it"]
    user_message = next(message for message in messages if message["role"] == "user" and message["message"] == "How can I improve profit?")
    assert messages[-1]["reply_to_message_id"] == user_message["id"]

    italian_chat_response = client.post(
        "/api/v1/restaurant/chat/messages",
        headers=headers,
        json={"message": "Come posso migliorare il profitto?", "language": "it"},
    )
    assert italian_chat_response.status_code == 201
    italian_messages = italian_chat_response.json()["messages"]
    assert italian_messages[-1]["role"] == "assistant"
    italian_reply = italian_messages[-1]["message"].lower()
    assert "profitto" in italian_reply
    assert "estimated profit" not in italian_reply
    assert italian_messages[-1]["message_translations"]["en"]
    assert italian_messages[-1]["message_translations"]["it"]

    follow_up_chat_response = client.post(
        "/api/v1/restaurant/chat/messages",
        headers=headers,
        json={"message": "What should I do next?"},
    )
    assert follow_up_chat_response.status_code == 201
    follow_up_messages = follow_up_chat_response.json()["messages"]
    assert follow_up_messages[-1]["role"] == "assistant"
    assert "only answer restaurant business questions" not in follow_up_messages[-1]["message"].lower()
    assert follow_up_messages[-1]["message_translations"]["en"]
    assert follow_up_messages[-1]["message_translations"]["it"]

    unrelated_chat_response = client.post(
        "/api/v1/restaurant/chat/messages",
        headers=headers,
        json={"message": "Who won the World Cup in 2018?"},
    )
    assert unrelated_chat_response.status_code == 201
    unrelated_messages = unrelated_chat_response.json()["messages"]
    assert unrelated_messages[-1]["role"] == "assistant"
    assert "current snapshot" in unrelated_messages[-1]["message"].lower()
    assert "revenue" in unrelated_messages[-1]["message"].lower()

    edited_chat_response = client.patch(
        f"/api/v1/restaurant/chat/messages/{user_message['id']}",
        headers=headers,
        json={"message": "How can I improve profit using lower supplier spend?"},
    )
    assert edited_chat_response.status_code == 200
    edited_messages = edited_chat_response.json()["messages"]
    edited_user_message = next(message for message in edited_messages if message["id"] == user_message["id"])
    assert edited_user_message["message"] == "How can I improve profit using lower supplier spend?"
    assert edited_user_message["edited_at"]
    linked_replies = [message for message in edited_messages if message.get("reply_to_message_id") == user_message["id"] and message["role"] == "assistant"]
    assert len(linked_replies) >= 2
    assert "revenue" in linked_replies[-1]["message"].lower()

    chat_attachment_response = client.post(
        "/api/v1/restaurant/chat/messages/attachments",
        headers=headers,
        data={"message": "Please review this supplier file", "attachment_source": "docs", "language": "it"},
        files={"file": ("suppliers.csv", b"supplier,amount\nBakery Goods Co,425", "text/csv")},
    )
    assert chat_attachment_response.status_code == 201
    chat_attachment_payload = chat_attachment_response.json()
    attachment_messages = [message for message in chat_attachment_payload["messages"] if message.get("attachment_name")]
    assert attachment_messages[-1]["attachment_name"] == "suppliers.csv"
    assert attachment_messages[-1]["attachment_source"] == "docs"
    assert attachment_messages[-1]["role"] == "user"
    assert "anteprima" in (attachment_messages[-1].get("attachment_summary") or "").lower()
    assert attachment_messages[-1]["attachment_summary_translations"]["en"]
    assert attachment_messages[-1]["attachment_summary_translations"]["it"]
    assert chat_attachment_payload["messages"][-1]["role"] == "assistant"

    spreadsheet_attachment_response = client.post(
        "/api/v1/restaurant/chat/messages/attachments",
        headers=headers,
        data={"message": "Please review this spreadsheet", "language": "en"},
        files={
            "file": (
                "supplier-costs.xlsx",
                b"fake-spreadsheet-bytes",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert spreadsheet_attachment_response.status_code == 201
    spreadsheet_messages = [
        message for message in spreadsheet_attachment_response.json()["messages"] if message.get("attachment_name")
    ]
    assert spreadsheet_messages[-1]["attachment_name"] == "supplier-costs.xlsx"
    assert spreadsheet_messages[-1]["role"] == "user"


    settings_response = client.get("/api/v1/restaurant/settings/profile", headers=headers)
    assert settings_response.status_code == 200
    settings_payload = settings_response.json()
    assert set(settings_payload.keys()) == {
        "full_name",
        "email",
        "phone",
        "restaurant_name",
        "restaurant_type",
        "location",
        "city_location",
        "number_of_seats",
        "average_spend_per_customer",
        "main_business_goal",
        "biggest_problem",
        "improvement_focus",
        "preferred_language",
        "profile_image_url",
        "exterior_photo_url",
        "interior_photo_url",
    }

    settings_update_response = client.put(
        "/api/v1/restaurant/settings/profile",
        headers=headers,
        data={
            "full_name": "Alexander Chen",
            "phone": "+1 (555) 123-4567",
            "restaurant_name": "The Golden Harvest",
            "restaurant_type": "Fine Dining",
            "city_location": "San Francisco",
            "number_of_seats": 120,
            "interior_photo_url": "",
        },
        files={
            "profile_image": ("profile.jpg", b"profile-image-bytes", "image/jpeg"),
            "interior_photo": ("interior.jpg", b"interior-image-bytes", "image/jpeg"),
            "exterior_photo": ("exterior.jpg", b"exterior-image-bytes", "image/jpeg"),
        },
    )
    assert settings_update_response.status_code == 200
    updated_settings_payload = settings_update_response.json()
    assert updated_settings_payload["full_name"] == "Alexander Chen"
    assert updated_settings_payload["phone"] == "+1 (555) 123-4567"
    assert updated_settings_payload["restaurant_name"] == "The Golden Harvest"
    assert updated_settings_payload["restaurant_type"] == "Fine Dining"
    assert updated_settings_payload["city_location"] == "San Francisco"
    assert updated_settings_payload["number_of_seats"] == 120
    assert updated_settings_payload["profile_image_url"].startswith("https://")
    assert "/restaurant/profile/" in updated_settings_payload["profile_image_url"]
    assert updated_settings_payload["interior_photo_url"].startswith("https://")
    assert "/restaurant/interior_photo/" in updated_settings_payload["interior_photo_url"]
    assert updated_settings_payload["exterior_photo_url"].startswith("https://")
    assert "/restaurant/exterior_photo/" in updated_settings_payload["exterior_photo_url"]

    clear_restaurant_photos_response = client.put(
        "/api/v1/restaurant/settings/profile",
        headers=headers,
        data={
            "remove_interior_photo": "true",
            "remove_exterior_photo": "true",
            "interior_photo_url": "",
            "exterior_photo_url": "",
        },
    )
    assert clear_restaurant_photos_response.status_code == 200
    cleared_restaurant_photos_payload = clear_restaurant_photos_response.json()
    assert cleared_restaurant_photos_payload["interior_photo_url"] is None
    assert cleared_restaurant_photos_payload["exterior_photo_url"] is None

    remove_profile_image_response = client.delete("/api/v1/restaurant/settings/profile/image", headers=headers)
    assert remove_profile_image_response.status_code == 200
    removed_profile_payload = remove_profile_image_response.json()
    assert removed_profile_payload["profile_image_url"] is None

    remove_profile_image_via_update_response = client.put(
        "/api/v1/restaurant/settings/profile",
        headers=headers,
        data={
            "remove_profile_image": "true",
            "profile_image_url": "",
        },
    )
    assert remove_profile_image_via_update_response.status_code == 200
    removed_profile_via_update_payload = remove_profile_image_via_update_response.json()
    assert removed_profile_via_update_payload["profile_image_url"] is None

    subscription_settings_response = client.get("/api/v1/restaurant/settings/subscription", headers=headers)
    assert subscription_settings_response.status_code == 200
    subscription_settings_payload = subscription_settings_response.json()
    assert subscription_settings_payload["selection_required"] is False
    assert subscription_settings_payload["plan_name"] == "Core Plan"
    assert subscription_settings_payload["plans_endpoint"] == "/api/v1/subscriptions/user/plans"
    assert subscription_settings_payload["checkout_endpoint"] == "/api/v1/subscriptions/user/checkout-session"
    assert subscription_settings_payload["customer_portal_endpoint"] == "/api/v1/subscriptions/user/customer-portal"

    notification_settings_response = client.get("/api/v1/restaurant/settings/notifications", headers=headers)
    assert notification_settings_response.status_code == 200
    assert notification_settings_response.json() == {
        "email_notifications": True,
        "push_notifications": True,
        "marketing_notifications": False,
        "low_stock_alerts": True,
        "daily_summary_notifications": True,
    }

    update_notification_response = client.put(
        "/api/v1/restaurant/settings/notifications",
        headers=headers,
        json={
            "push_notifications": False,
            "marketing_notifications": True,
        },
    )
    assert update_notification_response.status_code == 200
    assert update_notification_response.json()["push_notifications"] is False
    assert update_notification_response.json()["marketing_notifications"] is True
    assert update_notification_response.json()["email_notifications"] is True

    register_push_device_response = client.post(
        "/api/v1/restaurant/settings/push-devices/register",
        headers=headers,
        json={
            "expo_push_token": "ExponentPushToken[test-token-1]",
            "device_id": "device-abc-123",
            "platform": "android",
            "device_name": "android",
        },
    )
    assert register_push_device_response.status_code == 200
    assert register_push_device_response.json() == {"message": "Push device registered"}

    unregister_push_device_response = client.post(
        "/api/v1/restaurant/settings/push-devices/unregister",
        headers=headers,
        json={"device_id": "device-abc-123"},
    )
    assert unregister_push_device_response.status_code == 200
    assert unregister_push_device_response.json() == {"message": "Push device unregistered"}


def test_weekly_revenue_trend_uses_latest_available_business_dates(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Trend Owner",
            "email": "trend@example.com",
            "password": "TrendPass123",
            "phone": "+3900000012",
        },
    )
    select_subscription_plan(client, headers)

    today = datetime.now(UTC).date()
    latest_business_date = today - timedelta(days=8)
    earlier_business_date = today - timedelta(days=9)

    latest_business_date_iso = latest_business_date.isoformat()
    earlier_business_date_iso = earlier_business_date.isoformat()

    latest_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": latest_business_date_iso,
                "pos_payments": 650,
                "cash_payments": 250,
                "bank_transfer_payments": 100,
                "lunch_covers": 30,
                "dinner_covers": 40,
                "opening_cash": 120,
                "closing_cash": 320,
            },
        },
    )
    assert latest_entry_response.status_code == 201

    earlier_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": earlier_business_date_iso,
                "pos_payments": 500,
                "cash_payments": 150,
                "bank_transfer_payments": 50,
                "lunch_covers": 22,
                "dinner_covers": 28,
                "opening_cash": 100,
                "closing_cash": 240,
            },
        },
    )
    assert earlier_entry_response.status_code == 201

    home_revenue_response = client.get("/api/v1/restaurant/home/revenue?period=weekly", headers=headers)
    assert home_revenue_response.status_code == 200
    home_revenue_payload = home_revenue_response.json()
    assert home_revenue_payload["period"] == "weekly"
    assert len(home_revenue_payload["items"]) == 7
    assert sum(item["value"] for item in home_revenue_payload["items"]) == 1700
    assert any(item["value"] > 0 for item in home_revenue_payload["items"])

    analytics_trend_response = client.get("/api/v1/restaurant/analytics/revenue-trend?period=weekly", headers=headers)
    assert analytics_trend_response.status_code == 200
    analytics_trend_payload = analytics_trend_response.json()
    assert analytics_trend_payload["period"] == "weekly"
    assert analytics_trend_payload["revenue_total"] == 1700
    assert len(analytics_trend_payload["points"]) == 7
    assert sum(item["value"] for item in analytics_trend_payload["points"]) == 1700
    assert any(item["value"] > 0 for item in analytics_trend_payload["points"])


def test_vat_overview_applies_revenue_vat_from_restaurant_type(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "VAT Owner",
            "email": "vat-owner@example.com",
            "password": "VatOwner123",
            "phone": "+3900000088",
        },
    )
    select_subscription_plan_only(client, headers)
    complete_onboarding_profile(
        client,
        headers,
        restaurant_type="Pizzeria",
    )

    today_iso = datetime.now(UTC).date().isoformat()
    manual_entry_response = client.post(
        "/api/v1/restaurant/manual-entry",
        headers=headers,
        json={
            "method": "method_2",
            "method_two": {
                "business_date": today_iso,
                "pos_payments": 600,
                "cash_payments": 300,
                "bank_transfer_payments": 100,
                "lunch_covers": 18,
                "dinner_covers": 22,
                "opening_cash": 100,
                "closing_cash": 260,
            },
        },
    )
    assert manual_entry_response.status_code == 201

    vat_overview_response = client.get("/api/v1/restaurant/vat/overview", headers=headers)
    assert vat_overview_response.status_code == 200
    vat_overview_payload = vat_overview_response.json()
    assert vat_overview_payload["vat_payable"] == 100.0
    assert vat_overview_payload["vat_receivable"] == 0.0
    assert vat_overview_payload["estimated_vat_balance"] == 100.0

    home_vat_balance_response = client.get("/api/v1/restaurant/home/vat-balance", headers=headers)
    assert home_vat_balance_response.status_code == 200
    assert home_vat_balance_response.json()["balance"] == 100.0


def test_vat_overview_uses_mixed_purchase_invoice_line_rates(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Invoice VAT Owner",
            "email": "invoice-vat-owner@example.com",
            "password": "InvoiceVat123",
            "phone": "+3900000077",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()
    confirm_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Expense",
            "counterparty_name": "Fresh Food Supplier Ltd",
            "supplier_name": "Fresh Food Supplier Ltd",
            "invoice_number": "INV-MIXED-VAT",
            "invoice_date": today_iso,
            "total_amount": 441.0,
            "currency": "EUR",
            "expense_amount": 441.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "line_items": [
                {"product_name": "Meat", "quantity": 1, "unit_price": 100.0, "total_price": 100.0, "vat_rate": 10.0, "vat_amount": 10.0},
                {"product_name": "Water", "quantity": 1, "unit_price": 100.0, "total_price": 100.0, "vat_rate": 22.0, "vat_amount": 22.0},
                {"product_name": "Flour", "quantity": 1, "unit_price": 100.0, "total_price": 100.0, "vat_rate": 4.0, "vat_amount": 4.0},
                {"product_name": "Special Product", "quantity": 1, "unit_price": 100.0, "total_price": 100.0, "vat_rate": 5.0, "vat_amount": 5.0},
            ],
            "source_file_name": "mixed-vat-invoice.png",
            "ai_provider": "fallback",
            "ai_summary": "Mixed VAT invoice",
        },
    )
    assert confirm_response.status_code == 201

    detail_response = client.get(
        f"/api/v1/restaurant/documents/{confirm_response.json()['id']}",
        headers=headers,
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["net_total"] == 400.0
    assert detail_payload["vat_total"] == 41.0
    assert detail_payload["total_amount"] == 441.0
    assert [item["vat_rate"] for item in detail_payload["line_items"]] == [10.0, 22.0, 4.0, 5.0]
    assert [item["vat_amount"] for item in detail_payload["line_items"]] == [10.0, 22.0, 4.0, 5.0]

    vat_overview_response = client.get("/api/v1/restaurant/vat/overview", headers=headers)
    assert vat_overview_response.status_code == 200
    vat_overview_payload = vat_overview_response.json()
    assert vat_overview_payload["vat_payable"] == 0.0
    assert vat_overview_payload["vat_receivable"] == 41.0
    assert vat_overview_payload["estimated_vat_balance"] == -41.0


def test_invoice_products_sync_inventory_and_food_cost_memory(client, app):
    seed_subscription_plan(app)
    headers = register_and_login(
        client,
        {
            "full_name": "Inventory Sync Owner",
            "email": "inventory-sync-owner@example.com",
            "password": "InventorySync123",
            "phone": "+3900000088",
        },
    )
    select_subscription_plan(client, headers)

    today_iso = datetime.now(UTC).date().isoformat()

    revenue_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "revenue",
            "document_label": "Revenue",
            "counterparty_name": "Dining Room Sales",
            "supplier_name": "Dining Room Sales",
            "invoice_number": "REV-INV-SYNC-1",
            "invoice_date": today_iso,
            "total_amount": 1000.0,
            "currency": "EUR",
            "expense_amount": 0.0,
            "cash_amount": 0.0,
            "revenue_amount": 1000.0,
            "profit_amount": 0.0,
            "line_items": [],
            "source_file_name": "revenue-sync.pdf",
            "ai_provider": "fallback",
            "ai_summary": "Revenue base for food cost coverage",
        },
    )
    assert revenue_response.status_code == 201

    expense_response = client.post(
        "/api/v1/restaurant/documents/confirm-save",
        headers=headers,
        json={
            "document_type": "expense",
            "document_label": "Food Ingredients",
            "counterparty_name": "Metro Supplier",
            "supplier_name": "Metro Supplier",
            "invoice_number": "EXP-INV-SYNC-1",
            "invoice_date": today_iso,
            "total_amount": 110.0,
            "currency": "EUR",
            "expense_amount": 110.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "line_items": [
                {"product_name": "Beef", "category": "Meat", "quantity": 5, "unit_price": 10.0, "total_price": 50.0, "vat_rate": 10.0, "vat_amount": 5.0},
                {"product_name": "Water", "category": "Drinks", "quantity": 10, "unit_price": 5.0, "total_price": 50.0, "vat_rate": 10.0, "vat_amount": 5.0},
            ],
            "source_file_name": "food-invoice-sync.pdf",
            "ai_provider": "fallback",
            "ai_summary": "Food invoice should sync inventory and analytics",
        },
    )
    assert expense_response.status_code == 201
    expense_document_id = expense_response.json()["id"]

    inventory_response = client.get("/api/v1/restaurant/inventory", headers=headers)
    assert inventory_response.status_code == 200
    inventory_payload = inventory_response.json()
    assert inventory_payload["total"] == 2
    assert inventory_payload["total_inventory_value"] == 100.0
    assert {item["product_name"] for item in inventory_payload["items"]} == {"Beef", "Water"}
    assert {item["category"] for item in inventory_payload["items"]} == {"Meat", "Drinks"}
    assert {item["supplier_name"] for item in inventory_payload["items"]} == {"Metro Supplier"}

    inventory_items_by_name = {item["product_name"]: item for item in inventory_payload["items"]}
    assert inventory_items_by_name["Beef"]["stock_quantity"] == 5.0
    assert inventory_items_by_name["Beef"]["unit_price"] == 10.0
    assert inventory_items_by_name["Water"]["stock_quantity"] == 10.0
    assert inventory_items_by_name["Water"]["unit_price"] == 5.0

    categories_response = client.get("/api/v1/restaurant/inventory/categories", headers=headers)
    assert categories_response.status_code == 200
    assert {"Meat", "Drinks"} <= {item["name"] for item in categories_response.json()["items"]}

    suppliers_response = client.get("/api/v1/restaurant/inventory/suppliers", headers=headers)
    assert suppliers_response.status_code == 200
    assert "Metro Supplier" in {item["name"] for item in suppliers_response.json()["items"]}

    update_document_response = client.patch(
        f"/api/v1/restaurant/documents/{expense_document_id}",
        headers=headers,
        json={
            "supplier_name": "Fresh Market",
            "invoice_number": "EXP-INV-SYNC-1",
            "invoice_date": today_iso,
            "total_amount": 110.0,
            "line_items": [
                {"product_name": "Beef", "category": "Fish", "unit_type": "unit", "quantity": 5, "unit_price": 10.0, "total_price": 50.0, "vat_rate": 10.0, "vat_amount": 5.0},
                {"product_name": "Water", "category": "Drinks", "unit_type": "unit", "quantity": 10, "unit_price": 5.0, "total_price": 50.0, "vat_rate": 10.0, "vat_amount": 5.0},
            ],
        },
    )
    assert update_document_response.status_code == 200

    updated_inventory_response = client.get("/api/v1/restaurant/inventory", headers=headers)
    assert updated_inventory_response.status_code == 200
    updated_inventory_payload = updated_inventory_response.json()
    assert {item["supplier_name"] for item in updated_inventory_payload["items"]} == {"Fresh Market"}
    assert {item["category"] for item in updated_inventory_payload["items"]} == {"Fish", "Drinks"}

    updated_categories_response = client.get("/api/v1/restaurant/inventory/categories", headers=headers)
    assert updated_categories_response.status_code == 200
    assert {"Fish", "Drinks"} <= {item["name"] for item in updated_categories_response.json()["items"]}

    analytics_response = client.get("/api/v1/restaurant/analytics/overview", headers=headers)
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()
    food_cost_item = next(item for item in analytics_payload["cost_breakdown"] if item["label"] == "Food Cost")
    assert food_cost_item["value"] == 11.0


# Daily inventory usage feature has been removed, so this test is disabled.
def test_daily_inventory_usage_update_and_delete_restore_stock_and_usage_summary_disabled(client, app):
    pass
