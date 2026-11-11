/**
 * Next.js Route Handler: /api/admin/knowledge-gaps
 *
 * This server-side proxy forwards requests to the AI engine's knowledge-gap
 * endpoints after verifying that the requesting user is an admin.
 *
 * The AI_ADMIN_KEY env var is NEVER exposed to the browser.
 *
 * File path on VPS:
 *   /var/www/guapo-ai-frontend/app/api/admin/knowledge-gaps/route.ts
 */
import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

const AI_API = process.env.AI_API_URL ?? "";          // server-side only (no NEXT_PUBLIC_)
const AI_ADMIN_KEY = process.env.AI_ADMIN_KEY ?? "";  // server-side only
const BACKEND_API = process.env.NEXT_PUBLIC_API_URL ?? "";

// ---------------------------------------------------------------------------
// Auth helper — verifies JWT from cookie against the backend /users/me endpoint
// ---------------------------------------------------------------------------
async function getAdminUser(req: NextRequest): Promise<boolean> {
  // Try Authorization header first, then cookies
  let token =
    req.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";

  if (!token) {
    const cookieStore = await cookies();
    // Try common cookie names used by this application
    token =
      cookieStore.get("auth_token")?.value ??
      cookieStore.get("token")?.value ??
      "";
  }

  if (!token) return false;

  try {
    const res = await fetch(`${BACKEND_API}/users/me`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!res.ok) return false;
    const user = await res.json();
    return user?.role === "ADMIN" || user?.role === "SUPER_ADMIN";
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// GET /api/admin/knowledge-gaps  → list gaps (proxies query params)
// ---------------------------------------------------------------------------
export async function GET(req: NextRequest) {
  if (!(await getAdminUser(req))) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { searchParams } = new URL(req.url);
  const target = `${AI_API}/knowledge-gaps?${searchParams.toString()}`;

  try {
    const upstream = await fetch(target, {
      headers: { "X-Admin-Key": AI_ADMIN_KEY },
      cache: "no-store",
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    console.error("[knowledge-gaps proxy]", err);
    return NextResponse.json({ error: "Upstream error" }, { status: 502 });
  }
}
