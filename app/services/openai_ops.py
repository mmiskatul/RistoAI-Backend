from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, ClassVar

import httpx

from app.config.settings import get_settings
from app.core.exceptions import ValidationException

logger = logging.getLogger(__name__)


class OpenAIOperationsService:
    _CACHE_TTL_SECONDS: ClassVar[float] = 60.0
    _generation_cache: ClassVar[dict[str, tuple[float, Any]]] = {}
    _inflight_generations: ClassVar[dict[str, asyncio.Task[Any]]] = {}

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openai_api_key) and not self.settings.testing

    @staticmethod
    def _resolve_language(language: str | None) -> str:
        candidate = str(language or "").strip().lower()
        return "it" if candidate.startswith("it") else "en"

    @staticmethod
    def _build_attachment_note(*, summary: str | None, language: str) -> str:
        if not summary:
            return ""
        if language == "it":
            return f" Documento allegato: {summary}"
        return f" Attached document: {summary}"

    def _build_chat_fallback_reply(
        self,
        *,
        prompt: str,
        language: str,
        metrics_context: dict[str, Any],
        attachment_context: dict[str, Any] | None = None,
    ) -> str:
        contextual_reply = self._build_contextual_chat_fallback_reply(
            prompt=prompt,
            language=language,
            metrics_context=metrics_context,
            attachment_context=attachment_context,
        )
        if contextual_reply:
            return contextual_reply

        revenue = float(metrics_context.get("revenue_total", 0.0) or 0.0)
        expenses = float(metrics_context.get("expenses_total", 0.0) or 0.0)
        profit = revenue - expenses
        attachment_note = self._build_attachment_note(
            summary=(attachment_context or {}).get("summary"),
            language=language,
        )
        if language == "it":
            return (
                f"Panoramica attuale: ricavi â‚¬{revenue:,.2f}, spese â‚¬{expenses:,.2f}, "
                f"profitto stimato â‚¬{profit:,.2f}. "
                f"In base alla tua richiesta \"{prompt}\", ti consiglio di controllare la voce di costo piu alta, "
                f"verificare i giorni con ricavi piu bassi e riconciliare i movimenti di cassa."
                f"{attachment_note}"
        )
        return (
            f"Current snapshot: revenue â‚¬{revenue:,.2f}, expenses â‚¬{expenses:,.2f}, "
            f"estimated profit â‚¬{profit:,.2f}. "
            f"Based on your request \"{prompt}\", focus on the highest cost area, review the lowest-revenue days, "
            f"and reconcile cash movements."
            f"{attachment_note}"
        )

    @classmethod
    def _build_contextual_chat_fallback_reply(
        cls,
        *,
        prompt: str,
        language: str,
        metrics_context: dict[str, Any],
        attachment_context: dict[str, Any] | None = None,
    ) -> str:
        business_context = metrics_context.get("business_context")
        if not isinstance(business_context, dict):
            return ""

        prompt_text = str(prompt or "").lower()
        price_intent = any(token in prompt_text for token in ("overpriced", "price", "prices", "expensive", "market", "prezzo", "prezzi", "caro", "mercato"))
        if not price_intent:
            return ""

        line_items = cls._flatten_context_line_items(business_context)
        if not line_items:
            return ""

        target = cls._find_target_line_item(prompt_text=prompt_text, line_items=line_items)
        if not target:
            return ""

        product_name = str(target.get("product_name") or "this item")
        unit_price = float(target.get("unit_price", 0) or 0)
        quantity = float(target.get("quantity", 0) or 0)
        supplier_name = str(target.get("supplier_name") or "the supplier")
        invoice_number = str(target.get("invoice_number") or "the recent invoice")
        market_range = cls._match_market_reference(product_name, business_context)

        if not market_range or unit_price <= 0:
            if language == "it":
                return (
                    f"Ho trovato {product_name} nella fattura {invoice_number} da {supplier_name}, "
                    f"con prezzo unitario â‚¬{unit_price:,.2f}. Non ho un range affidabile per questo articolo nel contesto disponibile, "
                    "ma puoi confrontarlo con gli ultimi acquisti dello stesso prodotto e chiedere al fornitore una quotazione per volume."
                )
            return (
                f"I found {product_name} on invoice {invoice_number} from {supplier_name} at â‚¬{unit_price:,.2f} per unit. "
                "I do not have a reliable reference range for that exact item in the available context, but compare it against your last purchases "
                "for the same product and ask the supplier for a volume quote."
            )

        low = float(market_range.get("low", 0) or 0)
        high = float(market_range.get("high", 0) or 0)
        midpoint = (low + high) / 2 if high > 0 else 0
        variance = ((unit_price - midpoint) / midpoint * 100) if midpoint else 0
        status = "above" if unit_price > high else "below" if unit_price < low else "within"
        action = (
            "renegotiate or request two competing quotes"
            if status == "above"
            else "keep monitoring quality, yield, and portion cost"
        )
        if language == "it":
            status_it = "sopra" if status == "above" else "sotto" if status == "below" else "dentro"
            action_it = "rinegoziare o chiedere due preventivi concorrenti" if status == "above" else "monitorare qualita, resa e costo porzione"
            return (
                f"Ho analizzato {product_name} nella fattura {invoice_number} da {supplier_name}: "
                f"quantita {quantity:g}, prezzo unitario â‚¬{unit_price:,.2f}. "
                f"Il riferimento realistico per {market_range.get('unit', 'unita')} e circa â‚¬{low:,.2f}-â‚¬{high:,.2f}. "
                f"Questo prezzo e {status_it} il range, con scostamento di circa {variance:+.1f}% dal punto medio. "
                f"Raccomandazione: {action_it}. Nota: {market_range.get('note', 'range indicativo')}."
            )
        return (
            f"I analyzed {product_name} on invoice {invoice_number} from {supplier_name}: "
            f"quantity {quantity:g}, unit price â‚¬{unit_price:,.2f}. "
            f"A realistic reference range per {market_range.get('unit', 'unit')} is about â‚¬{low:,.2f}-â‚¬{high:,.2f}. "
            f"Your price is {status} that range, about {variance:+.1f}% versus the midpoint. "
            f"Recommendation: {action}. Note: {market_range.get('note', 'indicative range')}."
        )

    @staticmethod
    def _flatten_context_line_items(business_context: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for invoice in business_context.get("recent_invoices", []) or []:
            if not isinstance(invoice, dict):
                continue
            for line in invoice.get("line_items", []) or []:
                if isinstance(line, dict):
                    items.append(
                        {
                            **line,
                            "supplier_name": invoice.get("supplier_name"),
                            "invoice_number": invoice.get("invoice_number"),
                            "invoice_date": invoice.get("invoice_date"),
                        }
                    )
        for item in business_context.get("inventory_items", []) or []:
            if isinstance(item, dict):
                items.append(
                    {
                        "product_name": item.get("product_name"),
                        "quantity": item.get("stock_quantity"),
                        "unit_price": item.get("unit_price"),
                        "supplier_name": item.get("supplier_name"),
                        "invoice_number": "inventory record",
                        "invoice_date": item.get("purchase_date"),
                    }
                )
        return items

    @staticmethod
    def _find_target_line_item(*, prompt_text: str, line_items: list[dict[str, Any]]) -> dict[str, Any] | None:
        best_item: dict[str, Any] | None = None
        best_score = 0
        prompt_tokens = {token for token in prompt_text.replace("/", " ").replace("-", " ").split() if len(token) >= 3}
        for item in line_items:
            name = str(item.get("product_name") or "").lower()
            name_tokens = {token for token in name.replace("/", " ").replace("-", " ").split() if len(token) >= 3}
            score = len(prompt_tokens & name_tokens)
            if score > best_score:
                best_score = score
                best_item = item
        return best_item

    @staticmethod
    def _match_market_reference(product_name: str, business_context: dict[str, Any]) -> dict[str, Any] | None:
        normalized_name = product_name.lower()
        for item in business_context.get("market_reference_ranges", []) or []:
            if not isinstance(item, dict):
                continue
            keywords = item.get("product_keywords", [])
            if any(str(keyword).lower() in normalized_name for keyword in keywords):
                return item
        return None

    @staticmethod
    def _build_chat_scope_refusal(*, language: str) -> str:
        if language == "it":
            return (
                "Posso rispondere solo a domande legate alla gestione del ristorante, "
                "come ricavi, costi, coperti, documenti, inventario, cassa, personale e operazioni."
            )
        return (
            "I can only answer restaurant business questions, such as revenue, costs, covers, "
            "documents, inventory, cash flow, staff, and operations."
        )

    @classmethod
    def _restaurant_scope_keywords(cls) -> set[str]:
        return {
            "restaurant", "business", "revenue", "sales", "profit", "cost", "costs", "expense", "expenses", "margin",
            "invoice", "invoices", "document", "documents", "supplier", "suppliers", "inventory", "stock", "cash",
            "deposit", "deposits", "bank", "payment", "payments", "staff", "payroll", "labor", "menu", "pricing",
            "price", "prices", "food", "drink", "kitchen", "waste", "covers", "table", "tables", "booking",
            "bookings", "reservation", "reservations", "operations", "analytics", "insight", "insights", "vat", "tax",
            "receipt", "receipts", "pos", "customer", "customers", "guest", "guests", "service", "turnover",
            "ristorante", "ricavi", "profitto", "costi", "spese", "fattura", "fatture", "documento", "documenti",
            "fornitore", "fornitori", "inventario", "magazzino", "cassa", "deposito", "banca", "pagamento",
            "pagamenti", "personale", "lavoro", "coperti", "prenotazione", "prenotazioni", "operazioni", "analisi",
            "iva", "scontrino", "scontrini", "sprechi", "vendite", "cliente", "clienti", "cucina", "prezzo", "prezzi",
        }

    @classmethod
    def _contains_restaurant_scope_keywords(cls, value: str | None) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in cls._restaurant_scope_keywords())

    @classmethod
    def _is_restaurant_business_query(
        cls,
        *,
        prompt: str,
        recent_messages: Sequence[dict[str, Any]] | None = None,
        attachment_context: dict[str, Any] | None = None,
    ) -> bool:
        normalized_prompt = str(prompt or "").strip().lower()
        if not normalized_prompt:
            return False

        if cls._contains_restaurant_scope_keywords(normalized_prompt):
            return True

        attachment_summary = ""
        if attachment_context:
            attachment_summary = " ".join(
                str(attachment_context.get(key) or "")
                for key in ("title", "summary")
            )
        if cls._contains_restaurant_scope_keywords(attachment_summary):
            return True

        recent_messages = list(recent_messages or [])
        recent_transcript = "\n".join(
            str(item.get("message") or "")
            for item in recent_messages[-6:]
        )
        return cls._contains_restaurant_scope_keywords(recent_transcript)

    @classmethod
    def _build_chat_scope_context(
        cls,
        *,
        prompt: str,
        recent_messages: Sequence[dict[str, Any]] | None = None,
        attachment_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        recent_messages = list(recent_messages or [])
        attachment_summary = ""
        if attachment_context:
            attachment_summary = " ".join(
                str(attachment_context.get(key) or "")
                for key in ("title", "summary")
            ).strip()

        prompt_in_scope = cls._contains_restaurant_scope_keywords(prompt)
        attachment_in_scope = cls._contains_restaurant_scope_keywords(attachment_summary)
        recent_transcript = "\n".join(
            str(item.get("message") or "")
            for item in recent_messages[-6:]
        )
        recent_context_in_scope = cls._contains_restaurant_scope_keywords(recent_transcript)

        return {
            "prompt_in_scope": prompt_in_scope,
            "recent_context_in_scope": recent_context_in_scope,
            "attachment_in_scope": attachment_in_scope,
            "is_follow_up": not prompt_in_scope and (recent_context_in_scope or attachment_in_scope),
        }

    async def extract_invoice(
        self,
        *,
        file_name: str,
        content_type: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ValidationException("AI service is not configured or currently unavailable. Please check OpenAI API settings.")

        try:
            payload = await self._build_invoice_payload(
                file_name=file_name,
                content_type=content_type,
                file_bytes=file_bytes,
            )
            response_payload = await self._responses_create(payload)
            parsed = self._try_parse_json(response_payload.get("output_text", ""))
            if parsed:
                return parsed
        except httpx.HTTPStatusError as exc:
            logger.warning("OpenAI invoice extraction rejected request for %s (%s): %s", file_name, content_type, exc)
            raise ValidationException("AI service rejected the document processing request. Please ensure the image/document contains legible invoice details.") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI invoice extraction failed for %s", file_name, exc_info=exc)
            raise ValidationException("Unable to extract invoice data from this document using AI. Please try again with a clearer image.") from exc
        raise ValidationException("Failed to extract invoice data. The document could not be read properly by the AI service.")

    async def generate_business_insight(
        self,
        *,
        analytics_context: dict[str, Any],
        fallback_title: str,
        fallback_subtitle: str,
        language: str = "en",
    ) -> dict[str, str]:
        resolved_language = self._resolve_language(language)
        if not self.enabled:
            return {"title": fallback_title, "subtitle": fallback_subtitle, "ai_provider": "fallback"}

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You generate one concise restaurant analytics banner from structured business metrics. "
                                "Return only valid JSON with keys title and subtitle. "
                                f"The title must start with {'Optimization Tip:' if resolved_language == 'en' else 'Suggerimento di ottimizzazione:'}. "
                                "Be specific, numeric when possible, and grounded only in the supplied data. "
                                f"Return both fields in {'Italian' if resolved_language == 'it' else 'English'}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Analytics context: {json.dumps(analytics_context)}\n"
                                f"Fallback title: {fallback_title}\n"
                                f"Fallback subtitle: {fallback_subtitle}\n"
                                "Generate the strongest single optimization insight banner."
                            ),
                        }
                    ],
                },
            ],
        }
        cache_key = self._generation_cache_key(
            "business_insight",
            {
                "analytics_context": analytics_context,
                "fallback_title": fallback_title,
                "fallback_subtitle": fallback_subtitle,
            },
        )

        async def _generate() -> dict[str, str]:
            try:
                response_payload = await self._responses_create(payload)
                parsed = self._try_parse_json(response_payload.get("output_text", ""))
                if isinstance(parsed, dict):
                    title = str(parsed.get("title") or "").strip()
                    subtitle = str(parsed.get("subtitle") or "").strip()
                    if title and subtitle:
                        required_prefix = "Suggerimento di ottimizzazione:" if resolved_language == "it" else "Optimization Tip:"
                        if not title.startswith(required_prefix):
                            title = f"{required_prefix} {title}"
                        return {"title": title, "subtitle": subtitle, "ai_provider": "openai"}
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning("OpenAI business insight rate limited; using fallback insight.")
                else:
                    logger.exception("OpenAI business insight generation failed", exc_info=exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenAI business insight generation failed", exc_info=exc)
            return {"title": fallback_title, "subtitle": fallback_subtitle, "ai_provider": "fallback"}

        return await self._run_cached_generation(cache_key, _generate)

    async def generate_supplier_alerts(
        self,
        *,
        analytics_context: dict[str, Any],
        fallback_alerts: list[dict[str, str]],
        language: str = "en",
    ) -> list[dict[str, str]]:
        resolved_language = self._resolve_language(language)
        if not self.enabled:
            return [{**item, "ai_provider": "fallback"} for item in fallback_alerts]

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You generate concise supplier price alerts for a restaurant analytics dashboard from structured metrics. "
                                "Return only valid JSON with key alerts. "
                                "alerts must be an array of 1 to 3 objects, each with title and subtitle. "
                                "Keep each alert short, numeric when possible, and grounded only in the supplied data. "
                                f"Return all alert text in {'Italian' if resolved_language == 'it' else 'English'}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Supplier analytics context: {json.dumps(analytics_context)}\n"
                                f"Fallback alerts: {json.dumps(fallback_alerts)}\n"
                                "Generate the strongest supplier price alerts for the dashboard."
                            ),
                        }
                    ],
                },
            ],
        }
        cache_key = self._generation_cache_key(
            "supplier_alerts",
            {
                "analytics_context": analytics_context,
                "fallback_alerts": fallback_alerts,
            },
        )

        async def _generate() -> list[dict[str, str]]:
            try:
                response_payload = await self._responses_create(payload)
                parsed = self._try_parse_json(response_payload.get("output_text", ""))
                if isinstance(parsed, dict):
                    alerts = parsed.get("alerts")
                    if isinstance(alerts, list):
                        normalized: list[dict[str, str]] = []
                        for item in alerts[:3]:
                            if not isinstance(item, dict):
                                continue
                            title = str(item.get("title") or "").strip()
                            subtitle = str(item.get("subtitle") or "").strip()
                            if title and subtitle:
                                normalized.append({"title": title, "subtitle": subtitle, "ai_provider": "openai"})
                        if normalized:
                            return normalized
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning("OpenAI supplier alerts rate limited; using fallback alerts.")
                else:
                    logger.exception("OpenAI supplier alert generation failed", exc_info=exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenAI supplier alert generation failed", exc_info=exc)
            return [{**item, "ai_provider": "fallback"} for item in fallback_alerts]

        return await self._run_cached_generation(cache_key, _generate)

    async def generate_restaurant_insight(
        self,
        *,
        metrics_context: dict[str, Any],
        fallback_insight: dict[str, Any],
        language: str = "en",
    ) -> dict[str, Any]:
        resolved_language = self._resolve_language(language)
        if not self.enabled:
            return fallback_insight

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You generate the primary real-time restaurant operations insight for a mobile app. "
                                "Return only valid JSON with keys title, summary, priority, metric_value, metric_caption, "
                                "root_causes, and recommended_actions. "
                                "priority must be one of high, medium, low. "
                                "root_causes must be exactly 3 short strings. "
                                "recommended_actions must be exactly 3 objects with title and description. "
                                "Be specific, practical, and grounded only in the supplied metrics. "
                                f"Return every field in {'Italian' if resolved_language == 'it' else 'English'}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Metrics context: {json.dumps(metrics_context)}\n"
                                f"Fallback insight: {json.dumps(fallback_insight)}\n"
                                "Generate the strongest current insight that matches what a restaurant owner expects "
                                "to see in the frontend right now."
                            ),
                        }
                    ],
                },
            ],
        }
        cache_key = self._generation_cache_key(
            "restaurant_insight",
            {
                "metrics_context": metrics_context,
                "fallback_insight": fallback_insight,
            },
        )

        async def _generate() -> dict[str, Any]:
            try:
                response_payload = await self._responses_create(payload)
                parsed = self._try_parse_json(response_payload.get("output_text", ""))
                if isinstance(parsed, dict):
                    return self._normalize_restaurant_insight(parsed, fallback_insight)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning("OpenAI restaurant insight rate limited; using fallback insight.")
                else:
                    logger.exception("OpenAI restaurant insight generation failed", exc_info=exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenAI restaurant insight generation failed", exc_info=exc)
            return fallback_insight

        return await self._run_cached_generation(cache_key, _generate)

    @staticmethod
    def _normalize_restaurant_insight(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(fallback)
        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        priority = str(parsed.get("priority") or "").strip().lower()
        metric_value = str(parsed.get("metric_value") or "").strip()
        metric_caption = str(parsed.get("metric_caption") or "").strip()

        if title:
            normalized["title"] = title[:120]
        if summary:
            normalized["summary"] = summary[:240]
        if priority in {"high", "medium", "low"}:
            normalized["priority"] = priority
        if metric_value:
            normalized["metric_value"] = metric_value[:32]
        if metric_caption:
            normalized["metric_caption"] = metric_caption[:80]

        root_causes = parsed.get("root_causes")
        if isinstance(root_causes, list):
            causes = [str(item).strip()[:120] for item in root_causes if str(item).strip()]
            if causes:
                normalized["root_causes"] = (causes + fallback.get("root_causes", []))[:3]

        recommended_actions = parsed.get("recommended_actions")
        if isinstance(recommended_actions, list):
            actions: list[dict[str, str]] = []
            for item in recommended_actions:
                if not isinstance(item, dict):
                    continue
                action_title = str(item.get("title") or "").strip()
                action_description = str(item.get("description") or "").strip()
                if action_title and action_description:
                    actions.append(
                        {
                            "title": action_title[:80],
                            "description": action_description[:160],
                        }
                    )
            if actions:
                normalized["recommended_actions"] = (actions + fallback.get("recommended_actions", []))[:3]

        return normalized

    async def generate_chat_reply(
        self,
        *,
        prompt: str,
        language: str = "en",
        metrics_context: dict[str, Any],
        recent_messages: Sequence[dict[str, Any]],
        attachment_context: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> str:
        resolved_language = self._resolve_language(language)
        scope_context = self._build_chat_scope_context(
            prompt=prompt,
            recent_messages=recent_messages,
            attachment_context=attachment_context,
        )
        if not self._is_restaurant_business_query(
            prompt=prompt,
            recent_messages=recent_messages,
            attachment_context=attachment_context,
        ):
            return self._build_chat_scope_refusal(language=resolved_language)

        if not self.enabled:
            if resolved_language == "it":
                return (
                    "Il servizio AI non e al momento disponibile perche la chiave API non e configurata. "
                    "Contatta il tuo amministratore per abilitare il servizio AI."
                )
            return (
                "The AI service is currently unavailable because the API key is not configured. "
                "Please contact your administrator to enable the AI service."
            )

        transcript = "\n".join(f"{item['role']}: {item['message']}" for item in recent_messages[-6:])
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are an operations copilot for a restaurant back office app. "
                                "You must answer only restaurant business and operations questions. "
                                "If the user asks anything outside restaurant business scope, refuse briefly. "
                                "Treat short follow-up questions as in-scope when the recent conversation "
                                "or attachment context is clearly about restaurant operations. "
                                "Use the supplied metrics, business_context, recent invoices, invoice line items, inventory, suppliers, "
                                "account profile, market_reference_ranges, and long-term memory before giving advice. "
                                "Use a compact professional format when helpful: direct answer, evidence, and recommendation. "
                                "For broader requests, analyze relevant dimensions from available data: supplier pricing, margin, inventory risk, "
                                "cash flow, revenue/covers, waste, and operational bottlenecks. "
                                "Use memory for user preferences, recurring concerns, known suppliers/products, and previous goals, "
                                "but do not expose memory mechanics unless the user asks. "
                                "For supplier price questions, identify the invoice line item, compare its unit price to the supplied market range, "
                                "state whether it is below, within, or above range, and recommend a concrete action. "
                                "Do not tell the user to manually check data that is already present in context. "
                                "If exact market data is unavailable, say it is an indicative range and explain the assumption. "
                                "Keep replies concise, practical, numeric when possible, and grounded in the supplied data. "
                                f"Always answer in {'Italian' if resolved_language == 'it' else 'English'}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Metrics context: {json.dumps(metrics_context)}\n"
                                f"Long-term memory: {json.dumps(memory_context) if memory_context else 'none'}\n"
                                f"Recent messages:\n{transcript}\n"
                                f"Attachment context: {json.dumps(attachment_context) if attachment_context else "none"}\n"
                                f"Scope context: {json.dumps(scope_context)}\n"
                                f"User question: {prompt}"
                            ),
                        }
                    ],
                },
            ],
        }
        try:
            response_payload = await self._responses_create(payload)
            output_text = str(response_payload.get("output_text") or "").strip()
            if output_text:
                return output_text
            return self._build_chat_fallback_reply(
                prompt=prompt,
                language=resolved_language,
                metrics_context=metrics_context,
                attachment_context=attachment_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI chat generation failed", exc_info=exc)
            if resolved_language == "it":
                return (
                    "Il servizio AI ha riscontrato un errore temporaneo. "
                    "Per favore riprova tra qualche istante."
                )
            return (
                "The AI service encountered a temporary error. "
                "Please try again in a moment."
            )

    async def summarize_chat_memory(
        self,
        *,
        existing_memory: dict[str, Any] | None,
        recent_messages: Sequence[dict[str, Any]],
        business_context: dict[str, Any] | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        resolved_language = self._resolve_language(language)
        fallback = self._fallback_chat_memory_summary(
            existing_memory=existing_memory,
            recent_messages=recent_messages,
            business_context=business_context,
            language=resolved_language,
        )
        if not self.enabled:
            return fallback

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Return only valid JSON with keys summary, preferences, recurring_topics, "
                                "known_entities, and last_user_intent. Maintain a concise professional memory "
                                "for one restaurant account. Store durable business facts, user preferences, "
                                "recurring concerns, suppliers, products, and operational goals. Do not store "
                                "sensitive payment credentials or unrelated personal details."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Existing memory: {json.dumps(existing_memory) if existing_memory else 'none'}\n"
                                f"Recent messages: {json.dumps(list(recent_messages))}\n"
                                f"Business context: {json.dumps(business_context) if business_context else 'none'}\n"
                                f"Memory language: {'Italian' if resolved_language == 'it' else 'English'}"
                            ),
                        }
                    ],
                },
            ],
        }
        try:
            response_payload = await self._responses_create(payload)
            parsed = self._try_parse_json(response_payload.get("output_text", ""))
            if isinstance(parsed, dict):
                return self._normalize_chat_memory_payload(parsed, fallback=fallback)
            return fallback
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI chat memory summary failed", exc_info=exc)
            return fallback

    async def summarize_chat_attachment(
        self,
        *,
        file_name: str,
        content_type: str,
        file_bytes: bytes,
        language: str = "en",
    ) -> dict[str, str]:
        resolved_language = self._resolve_language(language)
        fallback = self._fallback_attachment_summary(
            file_name=file_name,
            content_type=content_type,
            file_bytes=file_bytes,
            language=resolved_language,
        )
        if not self.enabled:
            return fallback

        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Filename: {file_name}. Content type: {content_type}. "
                    f"Summarize this attachment for restaurant operations chat in "
                    f"{'Italian' if resolved_language == 'it' else 'English'}."
                ),
            }
        ]
        if content_type.startswith("image/"):
            data_url = f"data:{content_type};base64,{base64.b64encode(file_bytes).decode('utf-8')}"
            user_content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
        elif content_type in {"text/csv", "application/csv", "text/plain"}:
            user_content.append({"type": "input_text", "text": file_bytes.decode("utf-8", errors="replace")[:12000]})
        else:
            file_id = await self._upload_file(file_name=file_name, file_bytes=file_bytes)
            user_content.append({"type": "input_file", "file_id": file_id})

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Return only valid JSON with keys title and summary. "
                                "Summarize the attached restaurant-related document briefly and practically. "
                                f"Return the title and summary in {'Italian' if resolved_language == 'it' else 'English'}. "
                                "Do not invent missing facts."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }
        try:
            response_payload = await self._responses_create(payload)
            parsed = self._try_parse_json(response_payload.get("output_text", ""))
            if isinstance(parsed, dict):
                title = str(parsed.get("title") or "").strip()
                summary = str(parsed.get("summary") or "").strip()
                if title and summary:
                    return {"title": title, "summary": summary}
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI attachment summarization failed", exc_info=exc)
        return fallback

    async def translate_text(
        self,
        *,
        text: str,
        target_language: str = "en",
    ) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        resolved_language = self._resolve_language(target_language)
        if not self.enabled:
            return cleaned

        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Translate the provided restaurant app text accurately. "
                                "Preserve meaning, numbers, and business tone. "
                                "Return only the translated text with no explanation. "
                                f"Translate into {'Italian' if resolved_language == 'it' else 'English'}."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": cleaned,
                        }
                    ],
                },
            ],
        }
        cache_key = self._generation_cache_key(
            "translate_text",
            {
                "text": cleaned,
                "target_language": resolved_language,
            },
        )

        async def _generate() -> str:
            try:
                response_payload = await self._responses_create(payload)
                translated = str(response_payload.get("output_text") or "").strip()
                if translated:
                    return translated
            except Exception as exc:  # noqa: BLE001
                logger.exception("OpenAI text translation failed", exc_info=exc)
            return cleaned

        return await self._run_cached_generation(cache_key, _generate)

    async def transcribe_audio(
        self,
        *,
        file_name: str,
        content_type: str,
        file_bytes: bytes,
        language: str | None = None,
    ) -> str:
        if not file_bytes:
            return ""
        if not self.enabled:
            return ""

        form_data: dict[str, str] = {
            "model": self.settings.openai_transcription_model,
        }
        if language:
            form_data["language"] = language

        try:
            async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=60.0) as client:
                response = await client.post(
                    "/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    data=form_data,
                    files={"file": (file_name, file_bytes, content_type)},
                )
                response.raise_for_status()
                payload = response.json()
                return str(payload.get("text") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI audio transcription failed", exc_info=exc)
            return ""

    async def _responses_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=45.0) as client:
            headers = {
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            }
            for attempt in range(3):
                response = await client.post("/responses", headers=headers, json=payload)
                if response.status_code != 429:
                    response.raise_for_status()
                    data = response.json()
                    data["output_text"] = self._extract_output_text(data)
                    return data

                retry_after_header = response.headers.get("retry-after")
                try:
                    retry_after = float(retry_after_header) if retry_after_header else 0.0
                except ValueError:
                    retry_after = 0.0
                if attempt == 2:
                    response.raise_for_status()
                await asyncio.sleep(max(retry_after, 0.5 * (attempt + 1)))

            raise RuntimeError("OpenAI responses request retry loop exited unexpectedly")

    @classmethod
    def _generation_cache_key(cls, prefix: str, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    @classmethod
    async def _run_cached_generation(
        cls,
        cache_key: str,
        generator: Callable[[], Awaitable[Any]],
    ) -> Any:
        now = time.monotonic()
        cached = cls._generation_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

        inflight = cls._inflight_generations.get(cache_key)
        if inflight is not None:
            return await inflight

        task = asyncio.create_task(generator())
        cls._inflight_generations[cache_key] = task
        try:
            result = await task
            cls._generation_cache[cache_key] = (time.monotonic() + cls._CACHE_TTL_SECONDS, result)
            return result
        finally:
            current = cls._inflight_generations.get(cache_key)
            if current is task:
                cls._inflight_generations.pop(cache_key, None)

    async def _upload_file(self, *, file_name: str, file_bytes: bytes) -> str:
        async with httpx.AsyncClient(base_url=self.settings.openai_base_url, timeout=45.0) as client:
            response = await client.post(
                "/files",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                data={"purpose": "user_data"},
                files={"file": (file_name, file_bytes)},
            )
            response.raise_for_status()
            return response.json()["id"]

    async def _build_invoice_payload(
        self,
        *,
        file_name: str,
        content_type: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        system_prompt = (
            "Classify the uploaded restaurant finance document and extract it into JSON. "
            "Return only valid JSON with keys: "
            "document_type, document_label, counterparty_name, supplier_name, invoice_number, "
            "invoice_date, total_amount, currency, expense_amount, cash_amount, revenue_amount, "
            "profit_amount, ai_summary, and line_items. "
            "document_type must be one of expense, cash, revenue, profit, unknown. "
            "Use expense for supplier invoices, bills, and purchase receipts. "
            "Use cash for bank deposits, till counts, cash movement, or cash reconciliation documents. "
            "Use revenue for sales reports, receipts, POS summaries, or turnover documents. "
            "Use profit for P&L, income statement, margin report, or documents explicitly showing profit. "
            "Each line item must include product_name, quantity, unit_price, total_price, vat_rate, and vat_amount. "
            "Set vat_rate per product line, not once for the whole invoice. For Italian IVA, 4, 5, 10, and 22 are recommended common rates, "
            "but use any other percentage if it is clearly visible on the document. "
            "Use 10 for restaurant food items when the invoice does not show a rate. "
            "Use 0 for missing numeric fields and [] for missing line_items."
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Filename: {file_name}. Content type: {content_type}. "
                    "Extract the document classification and any expense, cash, revenue, or profit values."
                ),
            }
        ]

        if content_type.startswith("image/"):
            data_url = f"data:{content_type};base64,{base64.b64encode(file_bytes).decode('utf-8')}"
            user_content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
        elif content_type in {"text/csv", "application/csv"}:
            csv_text = file_bytes.decode("utf-8", errors="replace")
            user_content.append({"type": "input_text", "text": f"CSV invoice content:\n{csv_text[:12000]}"})
        else:
            file_id = await self._upload_file(file_name=file_name, file_bytes=file_bytes)
            user_content.append({"type": "input_file", "file_id": file_id})

        return {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

    @classmethod
    def _normalize_chat_memory_payload(cls, payload: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": str(payload.get("summary") or fallback.get("summary") or "").strip()[:1400],
            "preferences": cls._coerce_string_list(payload.get("preferences") or fallback.get("preferences"), limit=8),
            "recurring_topics": cls._coerce_string_list(payload.get("recurring_topics") or fallback.get("recurring_topics"), limit=10),
            "known_entities": cls._coerce_entity_list(payload.get("known_entities") or fallback.get("known_entities"), limit=15),
            "last_user_intent": str(payload.get("last_user_intent") or fallback.get("last_user_intent") or "").strip()[:240],
        }

    @staticmethod
    def _coerce_string_list(value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text[:160])
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _coerce_entity_list(value: Any, *, limit: int) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("value") or "").strip()
                entity_type = str(item.get("type") or item.get("entity_type") or "business").strip()
                note = str(item.get("note") or item.get("summary") or "").strip()
            else:
                name = str(item or "").strip()
                entity_type = "business"
                note = ""
            if name:
                result.append({"type": entity_type[:40], "name": name[:120], "note": note[:180]})
            if len(result) >= limit:
                break
        return result

    @classmethod
    def _fallback_chat_memory_summary(
        cls,
        *,
        existing_memory: dict[str, Any] | None,
        recent_messages: Sequence[dict[str, Any]],
        business_context: dict[str, Any] | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        previous_summary = str((existing_memory or {}).get("summary") or "").strip()
        user_messages = [
            str(item.get("message") or "").strip()
            for item in recent_messages
            if item.get("role") == "user" and str(item.get("message") or "").strip()
        ]
        last_user_intent = user_messages[-1][:240] if user_messages else str((existing_memory or {}).get("last_user_intent") or "")[:240]

        topics = cls._coerce_string_list((existing_memory or {}).get("recurring_topics"), limit=10)
        entities = cls._coerce_entity_list((existing_memory or {}).get("known_entities"), limit=15)
        combined_text = " ".join(user_messages).lower()
        topic_keywords = {
            "supplier pricing": ("price", "overpriced", "expensive", "market", "prezzo", "caro", "mercato"),
            "invoice analysis": ("invoice", "fattura", "uploaded", "document"),
            "inventory and costs": ("inventory", "stock", "cost", "margine", "scorte"),
            "cash flow": ("cash", "bank", "deposit", "reconcile", "cassa"),
        }
        for topic, keywords in topic_keywords.items():
            if topic not in topics and any(keyword in combined_text for keyword in keywords):
                topics.append(topic)

        if isinstance(business_context, dict):
            for invoice in business_context.get("recent_invoices", []) or []:
                if not isinstance(invoice, dict):
                    continue
                if invoice.get("supplier_name"):
                    entities.append({"type": "supplier", "name": str(invoice["supplier_name"])[:120], "note": "recent invoice supplier"})
                for line in invoice.get("line_items", []) or []:
                    if isinstance(line, dict) and line.get("product_name"):
                        entities.append({"type": "product", "name": str(line["product_name"])[:120], "note": "recent invoice item"})
            profile = business_context.get("restaurant_profile") if isinstance(business_context.get("restaurant_profile"), dict) else {}
            if profile.get("main_business_goal"):
                topics.append(str(profile["main_business_goal"])[:160])

        deduped_entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()
        for entity in entities:
            key = (entity.get("type", ""), entity.get("name", "").lower())
            if entity.get("name") and key not in seen_entities:
                seen_entities.add(key)
                deduped_entities.append(entity)

        new_summary_parts = [previous_summary] if previous_summary else []
        if last_user_intent:
            new_summary_parts.append(
                f"Recent focus: {last_user_intent}" if language != "it" else f"Focus recente: {last_user_intent}"
            )
        summary = " ".join(new_summary_parts).strip()[:1400]
        if not summary:
            summary = "Restaurant user prefers practical, data-grounded operational answers." if language != "it" else "L'utente preferisce risposte operative, pratiche e basate sui dati."

        return {
            "summary": summary,
            "preferences": cls._coerce_string_list((existing_memory or {}).get("preferences"), limit=8),
            "recurring_topics": topics[:10],
            "known_entities": deduped_entities[:15],
            "last_user_intent": last_user_intent,
        }

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        text_chunks: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_chunks.append(content.get("text", ""))
        return "\n".join(chunk for chunk in text_chunks if chunk).strip()

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _fallback_attachment_summary(
        *,
        file_name: str,
        content_type: str,
        file_bytes: bytes,
        language: str = "en",
    ) -> dict[str, str]:
        stem = file_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title() or 'Uploaded File'
        if content_type in {'text/csv', 'application/csv'}:
            preview = file_bytes.decode('utf-8', errors='replace').strip().splitlines()[:3]
            snippet = ' | '.join(line[:80] for line in preview if line)
            if language == "it":
                summary = (
                    f'Allegato CSV caricato. Anteprima: {snippet}'
                    if snippet else
                    'Allegato CSV caricato per la revisione.'
                )
            else:
                summary = f'CSV attachment uploaded. Preview: {snippet}' if snippet else 'CSV attachment uploaded for review.'
        elif content_type == 'text/plain':
            preview = file_bytes.decode('utf-8', errors='replace').strip()[:160]
            if language == "it":
                summary = (
                    f'Allegato di testo caricato. Anteprima: {preview}'
                    if preview else
                    'Allegato di testo caricato per la revisione.'
                )
            else:
                summary = f'Text attachment uploaded. Preview: {preview}' if preview else 'Text attachment uploaded for review.'
        elif content_type == 'application/pdf':
            summary = 'Documento PDF caricato per la revisione del ristorante.' if language == "it" else 'PDF document uploaded for restaurant review.'
        elif content_type.startswith('image/'):
            summary = 'Immagine caricata per la revisione del ristorante.' if language == "it" else 'Image attachment uploaded for restaurant review.'
        else:
            summary = 'Documento caricato per la revisione del ristorante.' if language == "it" else 'Document attachment uploaded for restaurant review.'
        return {'title': stem, 'summary': summary}

    @staticmethod
    def _fallback_invoice(*, file_name: str) -> dict[str, Any]:
        stem = file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        return {
            "document_type": "expense",
            "document_label": "Expense",
            "counterparty_name": "Fresh Food Supplier Ltd",
            "supplier_name": "Fresh Food Supplier Ltd",
            "invoice_number": "INV-2045",
            "invoice_date": datetime(2026, 3, 10, tzinfo=UTC).date().isoformat(),
            "total_amount": 165.0,
            "currency": "EUR",
            "expense_amount": 165.0,
            "cash_amount": 0.0,
            "revenue_amount": 0.0,
            "profit_amount": 0.0,
            "ai_summary": f"Fallback extraction generated for {stem}.",
            "line_items": [
                {"product_name": "Tomato Sauce", "quantity": 10, "unit_price": 5.0, "total_price": 50.0, "vat_rate": 10, "vat_amount": 5.0},
                {"product_name": "Cheese", "quantity": 5, "unit_price": 8.0, "total_price": 40.0, "vat_rate": 10, "vat_amount": 4.0},
                {"product_name": "Chicken", "quantity": 10, "unit_price": 6.0, "total_price": 60.0, "vat_rate": 10, "vat_amount": 6.0},
            ],
        }
