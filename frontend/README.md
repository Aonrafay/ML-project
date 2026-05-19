# SocialPulse AI — Frontend

Real-Time Misinformation & Trend Analyzer Dashboard

## Overview

The frontend is a **Next.js 14** application using the App Router, React, Tailwind CSS, and Chart.js to provide an interactive analytics dashboard for SocialPulse AI.

## Features

- **Live Feed** — Real-time stream of collected social media posts
- **Sentiment Timeline** — Sentiment analysis charts over time by topic
- **Trending Topics** — Dynamic topic visualization with engagement metrics
- **Post Explorer** — Searchable and filterable post browser
- **Fake/Real Classification Chart** — Distribution of fake vs. real news detection

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Next.js 14 (App Router) | Framework & SSR |
| React 18 | UI components |
| TypeScript | Type safety |
| Tailwind CSS | Styling |
| Chart.js + react-chartjs-2 | Data visualization |
| SWR / fetch | Data fetching |

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with providers
│   └── page.tsx            # Main dashboard page
├── components/
│   ├── FakeRealChart.tsx   # Fake/real classification chart
│   ├── LiveFeed.tsx        # Real-time post feed
│   ├── PostExplorer.tsx    # Searchable post table
│   ├── SentimentTimeline.tsx # Sentiment over time chart
│   └── TrendingTopics.tsx  # Trending topic cards
├── lib/
│   └ api.ts               # API client & helpers
├── styles/
│    globals.css           # Global styles + Tailwind
├── next.config.js         # Next.js configuration
├── tailwind.config.js     # Tailwind CSS configuration
├── tsconfig.json          # TypeScript configuration
└── package.json           # Dependencies & scripts
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

### Production Build

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_API_KEY=your_api_key_here
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api/v1` | Backend API base URL |
| `NEXT_PUBLIC_API_KEY` | — | API key for authentication |

## API Integration

All API calls are made through the [`lib/api.ts`](frontend/lib/api.ts) client module:

```typescript
import { fetchTrending, fetchSentiment, classifyText } from '@/lib/api';

// Fetch trending topics
const topics = await fetchTrending({ platform: 'reddit', limit: 10 });

// Get sentiment for a topic
const sentiment = await fetchSentiment('technology', { hours: 24 });

// Classify text
const result = await classifyText('This is a news article to verify');
```

### Available API Functions

| Function | Endpoint | Description |
|----------|----------|-------------|
| `fetchTrending()` | GET /trending | Fetch trending topics |
| `fetchTrendingRealtime()` | GET /trending/realtime | Real-time trending |
| `fetchSentiment()` | GET /sentiment/{topic} | Topic sentiment |
| `fetchOverallSentiment()` | GET /sentiment/overall | Overall sentiment |
| `classifyText()` | POST /classify | Classify text |
| `fetchFeed()` | GET /feed | Get post feed |
| `exportData()` | GET /export | Export data |

## Components

### FakeRealChart

Displays a bar/doughnut chart showing the distribution of fake vs. real news classifications.

```tsx
import FakeRealChart from '@/components/FakeRealChart';

<FakeRealChart data={classificationData} />
```

### LiveFeed

Shows a real-time scrolling feed of collected posts with platform badges and sentiment indicators.

```tsx
import LiveFeed from '@/components/LiveFeed';

<LiveFeed posts={recentPosts} />
```

### SentimentTimeline

Renders a line chart showing sentiment (positive/neutral/negative) over time for a given topic.

```tsx
import SentimentTimeline from '@/components/SentimentTimeline';

<SentimentTimeline topic="technology" hours={24} />
```

### TrendingTopics

Displays trending topic cards with engagement metrics and keyword lists.

```tsx
import TrendingTopics from '@/components/TrendingTopics';

<TrendingTopics topics={trendingData} />
```

### PostExplorer

A searchable, filterable table of posts with sorting and pagination.

```tsx
import PostExplorer from '@/components/PostExplorer';

<PostExplorer platform="reddit" limit={50} />
```

## Styling

The project uses **Tailwind CSS** for utility-first styling. Global styles and Tailwind directives are in [`styles/globals.css`](frontend/styles/globals.css).

Custom theme configuration is in [`tailwind.config.js`](frontend/tailwind.config.js).

## Docker

The frontend can be containerized using the production Dockerfile:

```bash
docker build -f deployment/Dockerfile.frontend -t socialpulse-frontend .
```

See [`deployment/docker-compose.prod.yml`](deployment/docker-compose.prod.yml) for full production deployment.

## Troubleshooting

| Issue | Solution |
|-------|---------|
| API connection errors | Verify backend is running and `NEXT_PUBLIC_API_URL` is correct |
| Chart rendering issues | Ensure `react-chartjs-2` and `chart.js` are installed |
| Tailwind not applying | Run `npm run build` to regenerate styles |
| TypeScript errors | Check `tsconfig.json` and ensure all types are defined |