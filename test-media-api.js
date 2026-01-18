/**
 * Test script for Media Analysis API
 * Tests the comprehensive media analysis functionality
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import the media analysis handler
import mediaAnalysisHandler from './api/media-analysis.js';

// Mock request and response objects
function createMockRequest(filePath) {
  const fileBuffer = fs.readFileSync(filePath);
  
  return {
    method: 'POST',
    body: {
      file: [{
        filepath: filePath,
        originalFilename: path.basename(filePath),
        size: fileBuffer.length,
        mimetype: 'video/mp4'
      }],
      analysisType: ['comprehensive']
    }
  };
}

function createMockResponse() {
  const response = {
    headers: {},
    statusCode: 200,
    data: null,
    
    setHeader(key, value) {
      this.headers[key] = value;
    },
    
    status(code) {
      this.statusCode = code;
      return this;
    },
    
    json(data) {
      this.data = data;
      console.log('\n🎬 MEDIA ANALYSIS RESULTS:');
      console.log('=====================================');
      console.log(JSON.stringify(data, null, 2));
      return this;
    },
    
    end() {
      return this;
    }
  };
  
  return response;
}

// Test function
async function testMediaAPI() {
  console.log('🚀 Testing Media Analysis API...\n');
  
  try {
    // Create a test video file (we'll use a small dummy file)
    const testVideoPath = path.join(__dirname, 'test-video.mp4');
    
    // Create a dummy video file for testing
    if (!fs.existsSync(testVideoPath)) {
      console.log('📹 Creating test video file...');
      // Create a small dummy file (in real scenario, you'd use an actual video)
      const dummyContent = Buffer.from('DUMMY_VIDEO_CONTENT_FOR_TESTING');
      fs.writeFileSync(testVideoPath, dummyContent);
    }
    
    // Create mock request and response
    const mockReq = createMockRequest(testVideoPath);
    const mockRes = createMockResponse();
    
    // Test the API
    console.log('🔍 Running comprehensive media analysis...');
    await mediaAnalysisHandler(mockReq, mockRes);
    
    // Analyze results
    if (mockRes.statusCode === 200 && mockRes.data) {
      console.log('\n✅ SUCCESS: Media API test completed!');
      console.log(`📊 APIs Attempted: ${mockRes.data.analysis?.apiSummary?.totalApis || 0}`);
      console.log(`✅ APIs Successful: ${mockRes.data.analysis?.apiSummary?.successfulApis || 0}`);
      console.log(`📈 Success Rate: ${mockRes.data.analysis?.apiSummary?.successRate || '0%'}`);
      console.log(`🎯 Risk Score: ${(mockRes.data.deepfakeInsights?.overallRiskScore * 100).toFixed(1)}%`);
    } else {
      console.log('❌ FAILED: API test failed');
      console.log('Status:', mockRes.statusCode);
      console.log('Data:', mockRes.data);
    }
    
  } catch (error) {
    console.error('❌ ERROR during testing:', error.message);
    console.log('\n💡 This is expected if API keys are not configured.');
    console.log('The API will gracefully handle missing credentials and provide fallback analysis.');
  }
  
  // Cleanup
  const testVideoPath = path.join(__dirname, 'test-video.mp4');
  if (fs.existsSync(testVideoPath)) {
    fs.unlinkSync(testVideoPath);
    console.log('\n🧹 Cleaned up test files');
  }
}

// Run the test
testMediaAPI();