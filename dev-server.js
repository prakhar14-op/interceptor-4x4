/**
 * Development server that handles both frontend and API routes
 * This mimics Vercel's behavior locally
 */

import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

// Enable CORS for all routes
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Import API handlers
import predictHandler from './api/predict.js';
import predictWithAgentsHandler from './api/predict-with-agents.js';
import predictLargeVideoHandler from './api/predict-large-video.js';
import chatAssistantHandler from './api/chat-assistant.js';
import mediaAnalysisHandler from './api/media-analysis.js';

// Middleware to convert Vercel API format to Express
const vercelToExpress = (handler) => {
  return async (req, res) => {
    try {
      // Set CORS headers
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

      if (req.method === 'OPTIONS') {
        return res.status(200).end();
      }

      await handler(req, res);
    } catch (error) {
      console.error('API Error:', error);
      res.status(500).json({ error: 'Internal server error', details: error.message });
    }
  };
};

// API Routes
app.post('/api/predict', vercelToExpress(predictHandler));
app.post('/api/predict-with-agents', vercelToExpress(predictWithAgentsHandler));
app.post('/api/predict-large-video', vercelToExpress(predictLargeVideoHandler));
app.post('/api/chat-assistant', vercelToExpress(chatAssistantHandler));
app.post('/api/media-analysis', vercelToExpress(mediaAnalysisHandler));

// Handle OPTIONS requests for all API routes
app.options('/api/*', (req, res) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.status(200).end();
});

// Proxy all other requests to Vite dev server
app.use('/', createProxyMiddleware({
  target: 'http://localhost:5173',
  changeOrigin: true,
  ws: true, // Enable WebSocket proxying for HMR
  onError: (err, req, res) => {
    console.error('Proxy error:', err);
    res.status(500).json({ error: 'Proxy error', details: err.message });
  }
}));

app.listen(PORT, () => {
  console.log(`🚀 Development server running on http://localhost:${PORT}`);
  console.log(`📡 API routes available at http://localhost:${PORT}/api/*`);
  console.log(`🎨 Frontend proxied from http://localhost:5173`);
  console.log(`\n🔧 Make sure Vite dev server is running on port 5173`);
  console.log(`\n💡 Now you can access your app at http://localhost:${PORT} with working APIs!`);
});