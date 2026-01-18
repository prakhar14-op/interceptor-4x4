/**
 * AI Assistant API with OpenAI GPT Integration
 * Provides intelligent analysis of deepfake detection results
 */

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { message, analysisData, conversationHistory } = req.body;

    // Try OpenAI API first, fallback to rule-based if not available
    let response;
    try {
      response = await generateOpenAIResponse(message, analysisData, conversationHistory);
    } catch (openaiError) {
      console.warn('OpenAI API unavailable, using fallback:', openaiError.message);
      response = generateFallbackResponse(message, analysisData);
    }

    return res.status(200).json({
      response,
      timestamp: new Date().toISOString(),
      source: response.includes('[AI-Generated]') ? 'openai' : 'fallback'
    });

  } catch (error) {
    console.error('Chat assistant error:', error);
    return res.status(500).json({ 
      error: 'Failed to process chat message',
      details: error.message 
    });
  }
}

/**
 * Generate response using OpenAI GPT API
 */
async function generateOpenAIResponse(message, analysisData, conversationHistory = []) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  
  if (!OPENAI_API_KEY) {
    throw new Error('OpenAI API key not configured');
  }

  // Build context for the AI
  const systemPrompt = `You are an expert AI assistant specializing in deepfake detection analysis. You help users understand video analysis results from a sophisticated deepfake detection system.

The system uses multiple specialist models:
- BG-Model-N (Background/Baseline): Detects general inconsistencies
- AV-Model-N (Audio-Visual): Analyzes audio-visual synchronization
- CM-Model-N (Compression): Detects compression artifacts (best performer, 70% accuracy)
- RR-Model-N (Resolution): Analyzes resolution inconsistencies
- LL-Model-N (Lighting): Detects lighting anomalies

Key metrics to explain:
- Confidence: 0-1 scale (>0.8 high, <0.6 low, 0.6-0.8 moderate)
- Model predictions: Individual model confidence scores
- Video characteristics: brightness, blur_score, resolution
- Processing details: frames analyzed, suspicious frames, heatmaps

Provide clear, helpful explanations that build user confidence in the results.`;

  const analysisContext = analysisData ? `
Current Analysis Data:
- Prediction: ${analysisData.prediction}
- Confidence: ${(analysisData.confidence * 100).toFixed(1)}%
- Models Used: ${analysisData.models_used?.join(', ') || 'Unknown'}
- Processing Time: ${analysisData.processing_time || 'Unknown'}s
- Video: ${analysisData.filename || 'Unknown'}
${analysisData.analysis ? `
- Video Quality: Brightness ${analysisData.analysis.routing?.video_characteristics?.brightness || 'Unknown'}, Blur ${analysisData.analysis.routing?.video_characteristics?.blur_score || 'Unknown'}
- Model Predictions: ${JSON.stringify(analysisData.analysis.model_predictions || {})}
- Suspicious Frames: ${analysisData.analysis.suspicious_frames || 0}
` : ''}` : 'No analysis data available yet.';

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'system', content: analysisContext },
    ...conversationHistory.slice(-10), // Keep last 10 messages for context
    { role: 'user', content: message }
  ];

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages: messages,
      max_tokens: 500,
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`OpenAI API error: ${error.error?.message || 'Unknown error'}`);
  }

  const data = await response.json();
  return `[AI-Generated] ${data.choices[0].message.content}`;
}

/**
 * Fallback response generator when OpenAI API is unavailable
 */
function generateFallbackResponse(message, analysisData) {
  const lowerMessage = message.toLowerCase();
  
  if (!analysisData) {
    return "I need analysis data to help you. Please upload a JSON analysis file first, or complete a video analysis in the workbench.";
  }

  // Enhanced rule-based responses
  if (lowerMessage.includes('confidence')) {
    const conf = analysisData.confidence * 100;
    const modelBreakdown = analysisData.analysis?.model_predictions ? 
      Object.entries(analysisData.analysis.model_predictions)
        .map(([model, score]) => `${model}: ${(score * 100).toFixed(1)}%`)
        .join(', ') : '';
    
    return `The overall confidence is ${conf.toFixed(1)}%. ${
      conf > 80 ? 'This is high confidence - the models are very certain about this prediction.' :
      conf < 60 ? 'This is low confidence - consider additional verification or manual review.' :
      'This shows moderate confidence in the result.'
    } ${modelBreakdown ? `Model breakdown: ${modelBreakdown}` : ''}`;
  }

  if (lowerMessage.includes('model') || lowerMessage.includes('algorithm')) {
    const models = analysisData.models_used || [];
    const predictions = analysisData.analysis?.model_predictions || {};
    let response = `This analysis used ${models.length} specialist models: ${models.join(', ')}. `;
    
    if (Object.keys(predictions).length > 0) {
      const bestModel = Object.entries(predictions).reduce((a, b) => 
        Math.abs(predictions[a[0]] - 0.5) > Math.abs(predictions[b[0]] - 0.5) ? a : b
      );
      response += `The ${bestModel[0]} model showed the strongest signal with ${(bestModel[1] * 100).toFixed(1)}% confidence.`;
    }
    
    return response;
  }

  if (lowerMessage.includes('fake') || lowerMessage.includes('real') || lowerMessage.includes('why')) {
    const prediction = analysisData.prediction;
    const confidence = analysisData.confidence;
    const reasons = [];
    
    if (analysisData.analysis?.model_predictions) {
      const models = analysisData.analysis.model_predictions;
      Object.entries(models).forEach(([model, conf]) => {
        if ((prediction === 'fake' && conf > 0.6) || (prediction === 'real' && conf < 0.4)) {
          reasons.push(`${model} model strongly supports this (${(conf * 100).toFixed(1)}%)`);
        }
      });
    }
    
    let response = `The video was classified as ${prediction.toUpperCase()} with ${(confidence * 100).toFixed(1)}% confidence. `;
    
    if (reasons.length > 0) {
      response += `Key supporting evidence: ${reasons.join(', ')}. `;
    }
    
    if (analysisData.analysis?.suspicious_frames > 0) {
      response += `${analysisData.analysis.suspicious_frames} suspicious frames were detected during analysis.`;
    }
    
    return response;
  }

  if (lowerMessage.includes('quality') || lowerMessage.includes('blur') || lowerMessage.includes('brightness')) {
    const chars = analysisData.analysis?.routing?.video_characteristics;
    if (chars) {
      let response = `Video quality analysis: `;
      response += `Resolution: ${chars.width}x${chars.height}, `;
      response += `Brightness: ${chars.brightness}, `;
      response += `Blur score: ${chars.blur_score}. `;
      
      const issues = [];
      if (chars.blur_score > 80) issues.push('high blur detected');
      if (chars.brightness < 50) issues.push('low brightness');
      if (chars.brightness > 200) issues.push('overexposed');
      
      if (issues.length > 0) {
        response += `Quality concerns: ${issues.join(', ')}. These factors may affect detection accuracy.`;
      } else {
        response += `Video quality appears acceptable for analysis.`;
      }
      
      return response;
    }
  }

  if (lowerMessage.includes('time') || lowerMessage.includes('process') || lowerMessage.includes('performance')) {
    const processingTime = analysisData.processing_time?.toFixed(2) || 'unknown';
    const framesAnalyzed = analysisData.analysis?.frames_analyzed || 'multiple';
    const heatmaps = analysisData.analysis?.heatmaps_generated || 0;
    
    return `Processing took ${processingTime} seconds. The analysis examined ${framesAnalyzed} frames and generated ${heatmaps} heatmaps for detailed inspection. ${
      parseFloat(processingTime) > 10 ? 'This was a thorough deep analysis.' : 'This was processed efficiently.'
    }`;
  }

  if (lowerMessage.includes('trust') || lowerMessage.includes('reliable') || lowerMessage.includes('accurate')) {
    const conf = analysisData.confidence;
    const modelCount = analysisData.models_used?.length || 0;
    
    let response = `Based on the analysis: `;
    
    if (conf > 0.8 && modelCount >= 3) {
      response += `High reliability - strong confidence (${(conf * 100).toFixed(1)}%) with multiple models (${modelCount}) in agreement.`;
    } else if (conf < 0.6) {
      response += `Lower reliability - consider additional verification due to moderate confidence (${(conf * 100).toFixed(1)}%).`;
    } else {
      response += `Moderate reliability - confidence is ${(conf * 100).toFixed(1)}% with ${modelCount} models analyzed.`;
    }
    
    if (analysisData.analysis?.routing?.video_characteristics?.blur_score > 80) {
      response += ` Note: High blur in the video may impact accuracy.`;
    }
    
    return response;
  }

  // Default contextual response
  const topics = [
    'confidence levels and what they mean',
    'which specialist models were used',
    'video quality factors affecting accuracy',
    'why the video was classified as fake or real',
    'processing performance and timing',
    'how reliable the results are'
  ];
  
  return `I can help explain various aspects of your ${analysisData.prediction} prediction (${(analysisData.confidence * 100).toFixed(1)}% confidence). Try asking about: ${topics.join(', ')}.`;
}