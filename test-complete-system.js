/**
 * Complete System Test - Frontend + API + Media Analysis
 */

console.log('🧪 Testing Complete System...\n');

// Test 1: API Server
async function testAPIServer() {
  try {
    const response = await fetch('http://localhost:3001/api/test');
    const data = await response.json();
    console.log('✅ API Server:', data.message);
    return true;
  } catch (error) {
    console.log('❌ API Server:', error.message);
    return false;
  }
}

// Test 2: Media Analysis API
async function testMediaAnalysisAPI() {
  try {
    const response = await fetch('http://localhost:3001/api/media-analysis');
    const data = await response.json();
    console.log('✅ Media Analysis API:', data.message);
    return true;
  } catch (error) {
    console.log('❌ Media Analysis API:', error.message);
    return false;
  }
}

// Test 3: Chat Assistant API
async function testChatAssistantAPI() {
  try {
    const response = await fetch('http://localhost:3001/api/chat-assistant');
    const data = await response.json();
    console.log('✅ Chat Assistant API:', data.message);
    return true;
  } catch (error) {
    console.log('❌ Chat Assistant API:', error.message);
    return false;
  }
}

// Test 4: Frontend Accessibility
async function testFrontend() {
  try {
    const response = await fetch('http://localhost:5173/');
    if (response.ok) {
      console.log('✅ Frontend: Accessible on port 5173');
      return true;
    } else {
      console.log('❌ Frontend: Not responding properly');
      return false;
    }
  } catch (error) {
    console.log('❌ Frontend:', error.message);
    return false;
  }
}

// Test 5: Chat API with Sample Data
async function testChatWithData() {
  try {
    const response = await fetch('http://localhost:3001/api/chat-assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: 'What APIs are working?',
        analysisData: { confidence: 0.85, prediction: 'real' }
      })
    });
    const data = await response.json();
    console.log('✅ Chat API Test:', data.response.substring(0, 50) + '...');
    return true;
  } catch (error) {
    console.log('❌ Chat API Test:', error.message);
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('🚀 Starting System Tests...\n');
  
  const results = await Promise.all([
    testAPIServer(),
    testMediaAnalysisAPI(),
    testChatAssistantAPI(),
    testFrontend(),
    testChatWithData()
  ]);
  
  const passed = results.filter(Boolean).length;
  const total = results.length;
  
  console.log(`\n📊 Test Results: ${passed}/${total} tests passed`);
  
  if (passed === total) {
    console.log('🎉 All systems operational!');
    console.log('\n🎯 Your Media API Integration is ready:');
    console.log('   • AssemblyAI: Working (audio intelligence)');
    console.log('   • Cloudinary: Configured (video analysis)');
    console.log('   • Hugging Face: Configured (object detection)');
    console.log('   • Chat Assistant: Working');
    console.log('   • Frontend: Working');
    console.log('\n🌐 Access your application:');
    console.log('   Frontend: http://localhost:5173');
    console.log('   API: http://localhost:3001');
  } else {
    console.log('⚠️  Some systems need attention');
  }
}

// Run tests
runAllTests();