# regime-trader dashboard

Static Next.js view of a published snapshot. It has **no credentials, no broker
connection, and no order path** — it reads `public/data/state.json` and nothing
else.

That is the whole design. The engine runs on your machine with your Alpaca keys;
this is a read-only window onto what it published. The alternative (serverless
routes calling Alpaca with keys in Vercel env vars) would mean a public URL that
can read a live account, and a dashboard holding broker handles is one bad deploy
away from being the reason the trading process died.

## Local

```bash
npm install
npm run dev            # http://localhost:3000
```

With no data it will tell you so. To get some:

```bash
cd ..
python main.py --publish-demo                    # labelled demo data
python main.py --dry-run --once --publish        # your own account
```

## Deploy

The repository root has a `vercel.json` that builds this directory, so import the
**repository root** into Vercel rather than this folder.

Because the site is a static export, the data is baked in at build time.
Refreshing the deployed dashboard means committing a new `state.json` and letting
Vercel redeploy:

```bash
python main.py --once --publish
git add dashboard/public/data/state.json && git commit -m "publish snapshot" && git push
```

For a live-updating view, use the terminal dashboard instead — it reads local
state directly and refreshes every 5 seconds:

```bash
python main.py --dashboard
```

## What ships in git

The committed `state.json` is **demo data**, labelled `"source": "demo"` and
rendered behind a banner. It exists so a fresh clone and the deployed URL show a
working interface rather than six empty panels. Publishing your own overwrites
it locally; think before committing a snapshot of a real account to a public
repository.

## Routing note

`vercel.json` sets `cleanUrls: true`. Next's static export writes a flat
`activity.html` rather than `activity/index.html`, so without it every route
except `/` returns 404 on Vercel while working fine under `next dev`. That gap
between local and deployed is the reason it is written down here.

The snapshot fetch in `lib/useSnapshot.ts` uses an absolute `/data/state.json`
for the related reason: a relative path resolves against the current route and
would ask for `/activity/data/state.json`.
