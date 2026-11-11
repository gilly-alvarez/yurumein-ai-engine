from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional
from pydantic import BaseModel
from app.core.config import settings
from ..utils.RAG_pipeline.DB.knowledge_gap_manager import (
    get_gap_manager,
    VALID_STATUSES,
    VALID_FLAG_REASONS,
)

router = APIRouter(prefix="/knowledge-gaps", tags=["Knowledge Gaps"])

# ---------------------------------------------------------------------------
# Server-side admin authentication
# ---------------------------------------------------------------------------

_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(key: Optional[str] = Security(_admin_key_header)) -> None:
    """FastAPI dependency: enforce X-Admin-Key header on all admin routes.

    Returns 503 when ADMIN_API_KEY is not configured (fail-secure default).
    Returns 403 for any missing or incorrect key.
    The key is passed server-side by the Next.js proxy — never from the browser.
    """
    configured = settings.ADMIN_API_KEY
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key not configured on this server.",
        )
    if not key or key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class GapUpdateRequest(BaseModel):
    status: Optional[str] = None
    admin_notes: Optional[str] = None
    source_reference: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints — all protected by require_admin_key
# ---------------------------------------------------------------------------


@router.get("", dependencies=[Depends(require_admin_key)])
async def list_gaps(
    status: Optional[str] = Query(None),
    flag_reason: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List knowledge gaps. Admin only — requires X-Admin-Key header."""
    if status and status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Valid values: {sorted(VALID_STATUSES)}",
        )
    if flag_reason and flag_reason not in VALID_FLAG_REASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flag_reason. Valid values: {sorted(VALID_FLAG_REASONS)}",
        )
    return await get_gap_manager().list_gaps(
        status=status, flag_reason=flag_reason, limit=limit, offset=offset
    )


@router.get("/report", dependencies=[Depends(require_admin_key)])
async def get_report():
    """On-demand weekly report. Admin only — requires X-Admin-Key header."""
    return await get_gap_manager().get_report()


@router.patch("/{gap_id}", dependencies=[Depends(require_admin_key)])
async def update_gap(gap_id: int, body: GapUpdateRequest):
    """Update status, notes, or source reference. Admin only — requires X-Admin-Key header."""
    if body.status and body.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Valid values: {sorted(VALID_STATUSES)}",
        )
    result = await get_gap_manager().update_gap(
        gap_id=gap_id,
        status=body.status,
        admin_notes=body.admin_notes,
        source_reference=body.source_reference,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Gap not found")
    return result


@router.delete("/{gap_id}", dependencies=[Depends(require_admin_key)])
async def delete_gap(gap_id: int):
    """Delete a gap record (privacy / retention control). Admin only — requires X-Admin-Key header."""
    deleted = await get_gap_manager().delete_gap(gap_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gap not found")
    return {"message": "Deleted", "id": gap_id}
