/**
 * Test Your Actual API Keys
 * Tests Cloudinary and AssemblyAI with your real credentials
 */

import dotenv from 'dotenv';
dotenv.config();

console.log('🧪 TESTING YOUR ACTUAL API CREDENTIALS');
console.log('=====================================\n');

// Test 1: Check if credentials are loaded
console.log('📋 Checking Environment Variables:');
console.log(`✅ Cloudinary Cloud Name: ${process.env.CLOUDINARY_CLOUD_NAME ? '✓ Set' : '❌ Missing'}`);
console.log(`✅ Cloudinary API Key: ${process.env.CLOUDINARY_API_KEY ? '✓ Set' : '❌ Missing'}`);
console.log(`✅ Cloudinary API Secret: ${process.env.CLOUDINARY_API_SECRET ? '✓ Set' : '❌ Missing'}`);
console.log(`✅ AssemblyAI API Key: ${process.env.ASSEMBLYAI_API_KEY ? '✓ Set' : '❌ Missing'}`);
console.log(`✅ Hugging Face API Key: ${process.env.HUGGINGFACE_API_KEY ? '✓ Set' : '❌ Missing (get from HF)'}\n`);

// Test 2: Test Cloudinary Connection
async function testCloudinary() {
  console.log('🎨 Testing Cloudinary Connection...');
  
  try {
    const cloudinary = (await import('cloudinary')).v2;
    
    cloudinary.config({
      cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
      api_key: process.env.CLOUDINARY_API_KEY,
      api_secret: process.env.CLOUDINARY_API_SECRET
    });

    // Test API connection
    const result = await cloudinary.api.ping();
    console.log('✅ Cloudinary: Connection successful!');
    console.log(`   Status: ${result.status}`);
    return true;
  } catch (error) {
    console.log('❌ Cloudinary: Connection failed');
    console.log(`   Error: ${error.message}`);
    return false;
  }
}

// Test 3: Test AssemblyAI Connection
async function testAssemblyAI() {
  console.log('\n🎤 Testing AssemblyAI Connection...');
  
  try {
    const response = await fetch('https://api.assemblyai.com/v2/transcript', {
      method: 'GET',
      headers: {
        'authorization': process.env.ASSEMBLYAI_API_KEY
      }
    });

    if (response.ok || response.status === 400) { // 400 is expected for GET without data
      console.log('✅ AssemblyAI: API key is valid!');
      console.log(`   Status: ${response.status}`);
      return true;
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    console.log('❌ AssemblyAI: Connection failed');
    console.log(`   Error: ${error.message}`);
    return false;
  }
}

// Test 4: Test Hugging Face Connection
async function testHuggingFace() {
  console.log('\n🤗 Testing Hugging Face Connection...');
  
  if (!process.env.HUGGINGFACE_API_KEY) {
    console.log('⏳ Hugging Face: API key not set yet');
    console.log('   Get your free key from: https://huggingface.co/settings/tokens');
    return false;
  }
  
  try {
    const response = await fetch('https://api-inference.huggingface.co/models/facebook/detr-resnet-50', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`
      }
    });

    if (response.ok) {
      console.log('✅ Hugging Face: API key is valid!');
      console.log(`   Status: ${response.status}`);
      return true;
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    console.log('❌ Hugging Face: Connection failed');
    console.log(`   Error: ${error.message}`);
    return false;
  }
}

// Run all tests
async function runAllTests() {
  const results = {
    cloudinary: await testCloudinary(),
    assemblyai: await testAssemblyAI(),
    huggingface: await testHuggingFace()
  };

  console.log('\n📊 TEST RESULTS SUMMARY:');
  console.log('========================');
  console.log(`🎨 Cloudinary: ${results.cloudinary ? '✅ WORKING' : '❌ FAILED'}`);
  console.log(`🎤 AssemblyAI: ${results.assemblyai ? '✅ WORKING' : '❌ FAILED'}`);
  console.log(`🤗 Hugging Face: ${results.huggingface ? '✅ WORKING' : '⏳ PENDING SETUP'}`);

  const workingApis = Object.values(results).filter(Boolean).length;
  console.log(`\n🎯 TOTAL: ${workingApis}/3 APIs working`);

  if (workingApis >= 2) {
    console.log('🎉 EXCELLENT! You have enough APIs for comprehensive media analysis!');
  } else if (workingApis >= 1) {
    console.log('👍 GOOD! You have basic media analysis capability!');
  } else {
    console.log('⚠️  Please check your API credentials and try again.');
  }

  console.log('\n💡 NEXT STEPS:');
  if (!results.huggingface) {
    console.log('1. Get your free Hugging Face API key: https://huggingface.co/settings/tokens');
  }
  if (workingApis > 0) {
    console.log('2. Test your media analysis with real video files!');
    console.log('3. Your comprehensive media API system is ready! 🚀');
  }
}

// Run the tests
runAllTests().catch(console.error);