/**
 * Simple Cloudinary Test
 */

import dotenv from 'dotenv';
dotenv.config();

console.log('🎨 CLOUDINARY SIMPLE TEST');
console.log('========================\n');

console.log('Credentials Check:');
console.log(`Cloud Name: "${process.env.CLOUDINARY_CLOUD_NAME}"`);
console.log(`API Key: "${process.env.CLOUDINARY_API_KEY}"`);
console.log(`API Secret: "${process.env.CLOUDINARY_API_SECRET ? 'SET' : 'MISSING'}"`);

// Test without importing cloudinary library
async function testCloudinaryAPI() {
  try {
    const auth = Buffer.from(`${process.env.CLOUDINARY_API_KEY}:${process.env.CLOUDINARY_API_SECRET}`).toString('base64');
    
    const response = await fetch(`https://api.cloudinary.com/v1_1/${process.env.CLOUDINARY_CLOUD_NAME}/resources/image`, {
      headers: {
        'Authorization': `Basic ${auth}`
      }
    });
    
    console.log(`\nAPI Response Status: ${response.status}`);
    
    if (response.ok) {
      console.log('✅ Cloudinary API is working!');
      return true;
    } else {
      const error = await response.text();
      console.log('❌ Cloudinary API failed:');
      console.log(`Error: ${error}`);
      return false;
    }
  } catch (error) {
    console.log('❌ Network error:', error.message);
    return false;
  }
}

testCloudinaryAPI();