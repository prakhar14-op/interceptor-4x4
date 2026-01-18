/**
 * Enhanced prediction endpoint that shows OnDemand agent integration
 * This endpoint simulates the full OnDemand workflow for testing
 */

export const config = {
  api: {
    bodyParser: false,
  },
};

import formidable from 'formidable';
import fs from 'fs';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    // Parse the uploaded file
    const form = formidable({ maxFileSize: 50 * 1024 * 1024, keepExtensions: true });
    const [fields, files] = await form.parse(req);
    const file = files.file?.[0];
    
    if (!file) return res.status(400).json({ error: 'No file uploaded' });

    const filename = file.originalFilename || 'test_video.mp4';
    const fileSize = file.size;

    // Simulate OnDemand agent analysis results
    const simulatedOnDemandData = {
      'llm-1': {
        output: `TECHNICAL ASSESSMENT for ${filename}:
- Resolution: 1280x720 detected
- Compression Level: medium quality  
- Frame Rate: 30fps consistent
- Bitrate Estimate: 1500 kbps
- Duration: ${Math.random() * 10 + 2} seconds

QUALITY METRICS:
- Overall Quality Score: ${(0.7 + Math.random() * 0.3).toFixed(2)}
- Artifacts Detected: minor compression artifacts
- Enhancement Needed: no

FORENSIC SUITABILITY:
- Legal Admissibility: suitable for analysis
- Chain of Custody: maintained
- Recommendations: proceed with deepfake detection`
      },
      'llm-2': {
        output: `METADATA ANALYSIS for ${filename}:
- File Size: ${fileSize} bytes
- Creation Timestamp: ${new Date().toISOString()}
- Modification History: no edits detected
- Codec Information: H.264/AAC

DEVICE INFORMATION:
- Recording Device: ${Math.random() > 0.5 ? 'iPhone 12 Pro' : 'Samsung Galaxy S21'}
- Software Used: native camera app
- GPS Coordinates: not present
- Device Settings: auto mode

INTEGRITY ASSESSMENT:
- Metadata Consistency: consistent
- Suspicious Indicators: none detected
- Chain of Custody: unbroken
- Forensic Hash: ${Math.random().toString(36).substring(2, 15)}`
      },
      'llm-3': {
        output: `CONTENT CLASSIFICATION for ${filename}:
- Content Type: human face detected
- Scene Setting: ${Math.random() > 0.5 ? 'indoor' : 'outdoor'}
- Lighting Conditions: ${Math.random() > 0.5 ? 'natural' : 'artificial'} lighting
- Background Type: ${Math.random() > 0.5 ? 'static' : 'dynamic'}

SUBJECT ANALYSIS:
- Face Count: ${Math.floor(Math.random() * 3) + 1} face(s) detected
- Audio Present: yes with good quality
- Motion Complexity: ${Math.random() > 0.5 ? 'low' : 'moderate'}
- Interaction Type: single person

LEGAL ASSESSMENT:
- Legal Relevance: high
- Privacy Concerns: none
- Evidence Quality: strong
- Classification Confidence: ${(0.8 + Math.random() * 0.2).toFixed(2)}`
      }
    };

    // Call our OnDemand webhook with the simulated data
    const protocol = req.headers.host?.includes('localhost') ? 'http' : 'https';
    const webhookUrl = `${protocol}://${req.headers.host}/api/ondemand-webhook`;
    
    console.log('Calling webhook:', webhookUrl);
    console.log('With data:', JSON.stringify(simulatedOnDemandData, null, 2));
    
    let webhookResponse;
    try {
      webhookResponse = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(simulatedOnDemandData)
      });
    } catch (fetchError) {
      console.error('Fetch error:', fetchError);
      throw new Error(`Failed to call webhook: ${fetchError.message}`);
    }

    console.log('Webhook response status:', webhookResponse.status);
    
    let webhookResult;
    try {
      const responseText = await webhookResponse.text();
      console.log('Webhook response text:', responseText);
      webhookResult = JSON.parse(responseText);
      console.log('Webhook result:', webhookResult);
    } catch (parseError) {
      console.error('Failed to parse webhook response:', parseError);
      throw new Error('Invalid response from webhook');
    }

    // Clean up temp file
    fs.unlinkSync(file.filepath);

    if (webhookResponse.ok && webhookResult && webhookResult.success && webhookResult.result) {
      // Enhance the result to show agent integration clearly
      const result = webhookResult.result;
      
      // Add agent models to models used
      if (!result.models_used.some(model => model.includes('Agent'))) {
        result.models_used.push(
          'Agent-1-Quality',
          'Agent-2-Metadata', 
          'Agent-3-Content'
        );
      }

      // Ensure ondemand_analysis is present
      if (!result.ondemand_analysis) {
        result.ondemand_analysis = {
          agents_used: 3,
          preprocessing_complete: true,
          agent_insights: {
            agent1: simulatedOnDemandData['llm-1'].output.substring(0, 200) + '...',
            agent2: simulatedOnDemandData['llm-2'].output.substring(0, 200) + '...',
            agent3: simulatedOnDemandData['llm-3'].output.substring(0, 200) + '...'
          },
          confidence_adjustment: Math.random() * 0.2 - 0.1
        };
      }

      return res.status(200).json({
        prediction: result.prediction,
        confidence: result.confidence,
        faces_analyzed: result.faces_analyzed,
        models_used: result.models_used,
        ondemand_analysis: result.ondemand_analysis,
        analysis: result.analysis,
        filename: result.filename,
        file_size: result.file_size,
        processing_time: result.processing_time,
        timestamp: result.timestamp,
        enhanced_by_agents: true
      });
    } else {
      console.error('Webhook failed:', webhookResult);
      return res.status(500).json({
        error: 'Webhook failed',
        details: webhookResult,
        webhook_status: webhookResponse.status
      });
    }

  } catch (error) {
    console.error('Agent prediction error:', error);
    res.status(500).json({
      error: 'Agent prediction failed',
      message: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
}