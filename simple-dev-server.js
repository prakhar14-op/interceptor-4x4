/**
 * Complete Development Server for API + Frontend
 */

import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();
const PORT = 3001;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

// Simple test endpoint
app.get('/api/test', (req, res) => {
  res.json({
    message: 'API server is working!',
    timestamp: new Date().toISOString(),
    status: 'success'
  });
});

// GET endpoint for media-analysis (for browser testing)
app.get('/api/media-analysis', (req, res) => {
  res.json({
    message: 'Media Analysis API is ready!',
    method: 'This endpoint accepts POST requests with video files',
    usage: 'Upload a video file via POST to analyze it',
    status: 'ready'
  });
});

// Media analysis endpoint (POST for actual analysis)
app.post('/api/media-analysis', (req, res) => {
  console.log('📹 Media analysis request received');
  
  // Simulate media analysis with your working AssemblyAI
  const mockResult = {
    success: true,
    analysis: {
      metadata: {
        filename: 'test-video.mp4',
        fileSize: 5242880,
        analysisTimestamp: new Date().toISOString()
      },
      apiSummary: {
        totalApis: 3,
        successfulApis: 1,
        successRate: "33.3%",
        apisUsed: ["assemblyai"]
      },
      audioAnalysis: {
        provider: 'AssemblyAI',
        transcriptionEnabled: true,
        contentSafety: true,
        status: 'working'
      }
    },
    deepfakeInsights: {
      overallRiskScore: 0.2,
      confidence: "low",
      recommendations: ["Audio analysis shows low risk of manipulation"]
    },
    summary: {
      processingTime: "2.1s"
    }
  };
  
  res.json(mockResult);
});

// GET endpoint for chat-assistant (for browser testing)
app.get('/api/chat-assistant', (req, res) => {
  res.json({
    message: 'Chat Assistant API is ready!',
    method: 'This endpoint accepts POST requests with messages',
    usage: 'Send a POST request with {"message": "your message"}',
    status: 'ready'
  });
});

// Chat assistant endpoint
app.post('/api/chat-assistant', (req, res) => {
  console.log('💬 Chat request received');
  
  const { message } = req.body;
  
  res.json({
    response: `I received your message: "${message}". Your chat API is working! Add OpenAI key for full functionality.`,
    timestamp: new Date().toISOString(),
    source: 'test-server'
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: '🎬 Interceptor Media Analysis API Server',
    status: 'running',
    endpoints: {
      'GET /api/test': 'Test API connectivity',
      'GET /api/media-analysis': 'Media analysis info',
      'POST /api/media-analysis': 'Analyze video files',
      'GET /api/chat-assistant': 'Chat assistant info', 
      'POST /api/chat-assistant': 'Chat with AI assistant'
    },
    frontend: 'Run "npm run dev" for frontend on port 5173',
    timestamp: new Date().toISOString()
  });
});

// Proxy frontend requests to Vite (if running)
app.use('*', createProxyMiddleware({
  target: 'http://localhost:5173',
  changeOrigin: true,
  ws: true,
  onError: (err, req, res) => {
    console.log('Frontend not available on port 5173');
    res.status(503).json({
      error: 'Frontend not available',
      message: 'Start the frontend with: npm run dev',
      api_server: 'API server is running on port 3001',
      endpoints: 'Visit http://localhost:3001 for API info'
    });
  }
}));

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Complete Dev Server running on http://localhost:${PORT}`);
  console.log(`📡 API endpoints:`);
  console.log(`   GET  http://localhost:${PORT}/api/test`);
  console.log(`   GET  http://localhost:${PORT}/api/media-analysis (info)`);
  console.log(`   POST http://localhost:${PORT}/api/media-analysis (upload)`);
  console.log(`   GET  http://localhost:${PORT}/api/chat-assistant (info)`);
  console.log(`   POST http://localhost:${PORT}/api/chat-assistant (chat)`);
  console.log(`\n🎨 Frontend:`);
  console.log(`   Start with: npm run dev`);
  console.log(`   Then visit: http://localhost:${PORT} (proxied to Vite)`);
  console.log(`\n💡 Your APIs are ready for testing!`);
});