/**
 * HTTP Test for Media Analysis API
 * Tests the API through actual HTTP requests
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test function using fetch
async function testMediaAPIHTTP() {
  console.log('🚀 Testing Media Analysis API via HTTP...\n');
  
  try {
    // Create a test video file
    const testVideoPath = path.join(__dirname, 'test-video.mp4');
    
    // Create a small test file
    console.log('📹 Creating test video file...');
    const dummyVideoContent = Buffer.alloc(1024 * 100, 'A'); // 100KB dummy file
    fs.writeFileSync(testVideoPath, dummyVideoContent);
    
    // Create FormData for the request
    const FormData = (await import('form-data')).default;
    const formData = new FormData();
    
    formData.append('file', fs.createReadStream(testVideoPath), {
      filename: 'test-video.mp4',
      contentType: 'video/mp4'
    });
    formData.append('analysisType', 'comprehensive');
    
    console.log('🔍 Sending request to Media Analysis API...');
    
    // Make the HTTP request
    const response = await fetch('http://localhost:5173/api/media-analysis', {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders()
    });
    
    console.log(`📡 Response Status: ${response.status}`);
    
    if (response.ok) {
      const result = await response.json();
      
      console.log('\n✅ SUCCESS: Media API HTTP test completed!');
      console.log('🎬 MEDIA ANALYSIS RESULTS:');
      console.log('=====================================');
      
      if (result.analysis) {
        console.log(`📊 APIs Attempted: ${result.analysis.apiSummary?.totalApis || 0}`);
        console.log(`✅ APIs Successful: ${result.analysis.apiSummary?.successfulApis || 0}`);
        console.log(`📈 Success Rate: ${result.analysis.apiSummary?.successRate || '0%'}`);
        console.log(`🎯 Risk Score: ${(result.deepfakeInsights?.overallRiskScore * 100).toFixed(1)}%`);
        console.log(`⏱️ Processing Time: ${result.summary?.processingTime || 'Unknown'}`);
        
        if (result.analysis.apiSummary?.apisUsed?.length > 0) {
          console.log(`🔌 APIs Used: ${result.analysis.apiSummary.apisUsed.join(', ')}`);
        }
        
        if (result.deepfakeInsights?.riskFactors?.length > 0) {
          console.log('\n🚨 Risk Factors Detected:');
          result.deepfakeInsights.riskFactors.forEach((factor, index) => {
            console.log(`  ${index + 1}. ${factor.factor} (${factor.severity} severity)`);
          });
        }
        
        if (result.deepfakeInsights?.recommendations?.length > 0) {
          console.log('\n💡 Recommendations:');
          result.deepfakeInsights.recommendations.forEach((rec, index) => {
            console.log(`  ${index + 1}. ${rec}`);
          });
        }
      } else {
        console.log('Raw Result:', JSON.stringify(result, null, 2));
      }
      
    } else {
      const errorText = await response.text();
      console.log('❌ FAILED: HTTP request failed');
      console.log('Status:', response.status);
      console.log('Error:', errorText);
      
      if (response.status === 404) {
        console.log('\n💡 TIP: The API endpoint might not be available in development.');
        console.log('This is expected if you\'re running only the Vite dev server.');
        console.log('The API routes need to be served by a backend server.');
      }
    }
    
  } catch (error) {
    console.error('❌ ERROR during HTTP testing:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      console.log('\n💡 CONNECTION REFUSED: The server might not be running on the expected port.');
      console.log('Make sure your development server is running and serving API routes.');
    }
  }
  
  // Cleanup
  const testVideoPath = path.join(__dirname, 'test-video.mp4');
  if (fs.existsSync(testVideoPath)) {
    fs.unlinkSync(testVideoPath);
    console.log('\n🧹 Cleaned up test files');
  }
}

// Test the API without external dependencies first
async function testAPILogic() {
  console.log('🧪 Testing API Logic (without external APIs)...\n');
  
  try {
    // Import our analysis functions
    const { generateComprehensiveReport, generateDeepfakeInsights } = await import('./api/media-analysis.js');
    
    // Create mock data
    const mockMediaAnalysis = {
      timestamp: new Date().toISOString(),
      filename: 'test-video.mp4',
      fileSize: 1024 * 100,
      mimeType: 'video/mp4',
      apis: {
        attempted: ['cloudinary', 'assemblyai', 'google-cloud'],
        successful: ['cloudinary'],
        failed: ['assemblyai', 'google-cloud']
      },
      results: {
        cloudinary: {
          provider: 'Cloudinary',
          videoSpecs: {
            duration: 30,
            width: 1920,
            height: 1080,
            frameRate: 30,
            format: 'mp4'
          },
          qualityMetrics: {
            qualityScore: 0.8,
            brightness: 120,
            contrast: 0.7
          },
          deepfakeIndicators: {
            faceConsistency: 0.9,
            compressionArtifacts: false
          }
        }
      }
    };
    
    const mockFile = {
      filepath: '/tmp/test-video.mp4',
      originalFilename: 'test-video.mp4',
      size: 1024 * 100
    };
    
    console.log('📊 Generating comprehensive report...');
    const report = generateComprehensiveReport(mockMediaAnalysis, mockFile);
    
    console.log('🎯 Generating deepfake insights...');
    const insights = generateDeepfakeInsights(report);
    
    console.log('\n✅ SUCCESS: API Logic test completed!');
    console.log('📈 Report generated with', Object.keys(report).length, 'sections');
    console.log('🎯 Risk Score:', (insights.overallRiskScore * 100).toFixed(1) + '%');
    console.log('🔍 Risk Factors:', insights.riskFactors.length);
    console.log('✅ Positive Indicators:', insights.positiveIndicators.length);
    
  } catch (error) {
    console.error('❌ ERROR in API logic test:', error.message);
  }
}

// Run both tests
console.log('🎬 COMPREHENSIVE MEDIA API TESTING');
console.log('===================================\n');

// Test 1: API Logic
await testAPILogic();

console.log('\n' + '='.repeat(50) + '\n');

// Test 2: HTTP API
await testMediaAPIHTTP();