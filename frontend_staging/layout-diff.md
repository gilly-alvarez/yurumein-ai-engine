# Frontend deployment instructions — Phase 1 Knowledge Gaps

## 1. New admin page

Copy `knowledge-gaps-page.tsx` to:
```
/var/www/guapo-ai-frontend/app/dashboard/(admin dashboard)/admin/knowledge-gaps/page.tsx
```

## 2. New Next.js Route Handlers (server-side proxy)

Create these directories and files so the AI admin key stays server-side:

```
/var/www/guapo-ai-frontend/app/api/admin/knowledge-gaps/route.ts
  ← copy api-knowledge-gaps-route.ts

/var/www/guapo-ai-frontend/app/api/admin/knowledge-gaps/report/route.ts
  ← copy api-knowledge-gaps-report-route.ts

/var/www/guapo-ai-frontend/app/api/admin/knowledge-gaps/[id]/route.ts
  ← copy api-knowledge-gaps-id-route.ts
```

## 3. Sidebar entry

In `/var/www/guapo-ai-frontend/app/dashboard/layout.tsx`, find the `adminSidebarItems` array and add after the existing "Lexicon" item:

```ts
{
  name: "Knowledge Gaps",
  href: "/dashboard/admin/knowledge-gaps",
  icon: "🔍",
},
```

## 4. Environment variables (frontend — server-side only, NO NEXT_PUBLIC_ prefix)

Add to `/var/www/guapo-ai-frontend/.env.local` (or the equivalent production env file):

```
# Server-side only — NEVER use NEXT_PUBLIC_ prefix for these
AI_API_URL=http://localhost:8085           # internal URL to AI engine (from Next.js server)
AI_ADMIN_KEY=<same value as ADMIN_API_KEY in AI engine .env>
```

Important:
- `AI_API_URL` (without NEXT_PUBLIC_) is the internal URL to the AI engine, accessible only from the Next.js server process (not from the browser).
- `AI_ADMIN_KEY` must match `ADMIN_API_KEY` in the AI engine's `.env`.
- The existing `NEXT_PUBLIC_AI_API_URL` remains for the existing feed-ai and other browser-side calls. The new knowledge-gap routes use the server-side `AI_API_URL` instead.

## 5. Build

```bash
cd /var/www/guapo-ai-frontend
npm run build
systemctl restart yurumein-frontend    # or pm2 restart, depending on the process manager
```

## 6. Verify

After deploy, navigate to:
  https://yurumeinai.com/dashboard/admin/knowledge-gaps

Confirm:
- Page loads and shows "Review Queue" and "Weekly Report" tabs
- A non-admin user cannot access the page (redirected by dashboard layout)
- An API call to GET /knowledge-gaps without X-Admin-Key returns 403
- An API call to GET /knowledge-gaps with correct X-Admin-Key returns JSON
