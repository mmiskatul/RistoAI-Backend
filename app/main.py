from __future__ import annotations

import utils.clients
import utils.http_session
import utils.transport
import utils.session
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pymongo.errors import PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.config.settings import get_settings
from app.demo_pages import landing_page_html
from app.core.constants import API_V1_PREFIX
from app.core.exceptions import AppException
from app.core.logging import configure_logging
from app.db.indexes import ensure_indexes
from app.db.migrations import run_data_migrations
from app.db.mongodb import MongoDB
from app.middleware.request_context import RequestContextMiddleware
from app.middleware.subscription_guard import SubscriptionGuardMiddleware
from app.realtime.socketio_app import create_socketio_app
from app.repositories.subscription_plan import SubscriptionPlanRepository
from app.repositories.user import UserRepository
from app.services.bootstrap import BootstrapService

logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {'name': 'Restaurant Authentication', 'description': 'Restaurant account registration, verification, login, and password reset.'},
    {'name': 'Authentication', 'description': 'Shared token and current-user identity endpoints.'},
    {'name': 'Restaurant Home', 'description': 'Mobile app home dashboard, summary cards, quick actions, and recent activity.'},
    {'name': 'Restaurant VAT', 'description': 'VAT overview, balances, payable, receivable, and filing summary.'},
    {'name': 'Restaurant Insights', 'description': 'Insight detail pages with business causes, charts, and recommended actions.'},
    {'name': 'Restaurant Invoice AI', 'description': 'OpenAI-assisted invoice upload, extraction preview, confirmation, saved invoice list, detail, update, and delete.'},
    {'name': 'Restaurant Expenses', 'description': 'Manual expense creation and expense list with summary cards.'},
    {'name': 'Restaurant Cash Management', 'description': 'Cash overview and bank deposit records.'},
    {'name': 'Restaurant Invoice Manual Entry', 'description': 'Manual invoice-style daily business entry create and update endpoints for method 1 and method 2 flows.'},
    {'name': 'Restaurant Data Management', 'description': 'Grouped date, week, and month records, drilldowns, and daily record detail views.'},
    {'name': 'Restaurant Analytics', 'description': 'Analytics overview, AI business insight banner, trend cards, comparison rows, and alerts.'},
    {'name': 'Restaurant Chat', 'description': 'Restaurant AI chat conversation read and send endpoints.'},
    {'name': 'Restaurant Settings', 'description': 'Restaurant profile and settings detail/update endpoints.'},
    {'name': 'Restaurant Support', 'description': 'Restaurant support ticket creation and personal ticket history.'},
    {'name': 'Onboarding', 'description': 'Restaurant onboarding profile and setup progress.'},
    {'name': 'User Subscription', 'description': 'Current subscription and subscription selection for restaurant users.'},
    {'name': 'Admin Authentication', 'description': 'Admin login and password reset endpoints.'},
    {'name': 'Admin Dashboard', 'description': 'Admin dashboard overview, KPI cards, revenue growth, and user growth metrics.'},
    {'name': 'Users', 'description': 'Admin user management and filtering.'},
    {'name': 'Subscription Management', 'description': 'Admin subscription plan, coupon, and revenue management.'},
    {'name': 'Support Management', 'description': 'Admin support ticket list, replies, and resolution.'},
    {'name': 'Settings', 'description': 'Admin platform settings and legal content editor endpoints.'},
    {'name': 'Health', 'description': 'Service health check.'},
]



def _make_json_safe(value: Any) -> Any:
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return jsonable_encoder(value)



def _error_response(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={'success': False, 'error': {'code': code, 'message': message, 'details': details or {}}},
    )



def _http_error_code(status_code: int) -> str:
    return {
        400: 'bad_request',
        401: 'unauthorized',
        403: 'forbidden',
        404: 'not_found',
        405: 'method_not_allowed',
        409: 'conflict',
        422: 'validation_error',
    }.get(status_code, 'http_error')



def _split_http_detail(detail: Any) -> tuple[str, dict[str, Any]]:
    safe_detail = _make_json_safe(detail)
    if isinstance(safe_detail, str):
        return safe_detail, {}
    return 'Request failed', {'detail': safe_detail}



def create_fastapi_app(*, testing: bool = False) -> FastAPI:
    settings = get_settings()
    settings.testing = testing
    if testing:
        settings.debug = True
        settings.smtp_enabled = False
        settings.resend_enabled = False
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.testing:
            db = MongoDB.connect(settings)
            await ensure_indexes(db)
            await run_data_migrations(db)
            bootstrap_service = BootstrapService(UserRepository(db), SubscriptionPlanRepository(db))
            await bootstrap_service.ensure_super_admin(settings)
            await bootstrap_service.ensure_default_subscription_plan(settings)
            logger.info('MongoDB connected, indexes ensured, data migrations completed, super admin synchronized, and default subscription plan ensured')
        yield
        if not settings.testing:
            MongoDB.close()
            logger.info('MongoDB connection closed')

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        openapi_url=settings.openapi_url,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SubscriptionGuardMiddleware)
    app.include_router(api_router, prefix=API_V1_PREFIX)

    @app.exception_handler(AppException)
    async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, _make_json_safe(exc.details))

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message, details = _split_http_detail(exc.detail)
        return _error_response(exc.status_code, _http_error_code(exc.status_code), message, details)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            422,
            'validation_error',
            'Request validation failed',
            {'errors': _make_json_safe(exc.errors())},
        )

    @app.exception_handler(ResponseValidationError)
    async def response_validation_exception_handler(_: Request, exc: ResponseValidationError) -> JSONResponse:
        logger.exception('Response validation failed', exc_info=exc)
        return _error_response(
            500,
            'response_validation_error',
            'Response validation failed',
            {'errors': _make_json_safe(exc.errors())},
        )

    @app.exception_handler(PyMongoError)
    async def mongo_exception_handler(_: Request, exc: PyMongoError) -> JSONResponse:
        logger.exception('Database operation failed', exc_info=exc)
        return _error_response(503, 'database_error', 'Database operation failed')

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception('Unhandled server error', exc_info=exc)
        return _error_response(500, 'internal_server_error', 'Unexpected server error')

    @app.get('/', include_in_schema=False)
    async def root() -> dict:
        return {
            'success': True,
            'message': f'{settings.app_name} is running',
            'version': settings.app_version,
            'docs': '/docs',
        }

    @app.get('/health', tags=['Health'])
    async def healthcheck() -> dict[str, Any]:
        database_connected = True if settings.testing else await MongoDB.ping(settings)
        return {
            'status': 'ok' if database_connected else 'degraded',
            'service': settings.app_name,
            'database': 'connected' if database_connected else 'unavailable',
        }

    @app.get('/preview/landing', response_class=HTMLResponse, include_in_schema=False)
    async def landing_preview() -> HTMLResponse:
        return HTMLResponse(content=landing_page_html())

    return app


def create_app(*, testing: bool = False):
    fastapi_app = create_fastapi_app(testing=testing)
    if testing:
        return fastapi_app
    return create_socketio_app(fastapi_app)


app = create_app()




