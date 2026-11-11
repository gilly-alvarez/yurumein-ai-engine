/**
 * Next.js Route Handler: /api/admin/knowledge-gaps/[id]
 *
 * File path on VPS:
 *   /var/www/guapo-ai-frontend/app/api/admin/knowledge-gaps/[id]/route.ts
 */
import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

const AI_API = process.env.AI_API_URL ?? "";
const AI_ADMIN_KEY = process.env.AI_ADMIN_KEY ?? "";
const BACKEND_API = process.env.NEXT_PUBLIC_API_URL ?? "";

async function getAdminUser(req: NextRequest): Promise<boolean> {
  let token =
    req.headers.get("authorization")?.replace(/^Bearer\s+/i, "") ?? "";
  if (!token) {
    const cookieStore = await cookies();
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

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await getAdminUser(req))) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  const { id } = await params;
  const body = await req.json();

  try {
    const upstream = await fetch(`${AI_API}/knowledge-gaps/${id}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": AI_ADMIN_KEY,
      },
      body: JSON.stringify(body),
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    console.error("[knowledge-gap PATCH proxy]", err);
    return NextResponse.json({ error: "Upstream error" }, { status: 502 });
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await getAdminUser(req))) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  const { id } = await params;

  try {
    const upstream = await fetch(`${AI_API}/knowledge-gaps/${id}`, {
      method: "DELETE",
      headers: { "X-Admin-Key": AI_ADMIN_KEY },
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    console.error("[knowledge-gap DELETE proxy]", err);
    return NextResponse.json({ error: "Upstream error" }, { status: 502 });
  }
}
