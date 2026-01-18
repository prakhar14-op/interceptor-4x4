import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, FileText, Brain, Zap, MessageCircle, Trash2, Download } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { ScrollArea } from '../components/ui/scroll-area';
import { Badge } from '../components/ui/badge';
import { Separator } from '../components/ui/separator';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysisData?: any;
}

interface AnalysisInsight {
  category: string;
  insight: string;
  confidence: number;
  recommendation?: string;
}

const AIAssistant = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: "Hello! I'm your AI Assistant for deepfake analysis. I can help you understand your video analysis results in detail. Upload a JSON analysis file or paste your analysis data to get started!",
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<any>(null);
  const [insights, setInsights] = useState<AnalysisInsight[]>([]);
  const [mediaAnalysis, setMediaAnalysis] = useState<any>(null);
  const [isAnalyzingMedia, setIsAnalyzingMedia] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check for analysis data from localStorage (from AnalysisWorkbench)
    const storedAnalysis = localStorage.getItem('currentAnalysis');
    if (storedAnalysis) {
      try {
        const analysisData = JSON.parse(storedAnalysis);
        analyzeJsonData(analysisData);
        localStorage.removeItem('currentAnalysis'); // Clean up
      } catch (error) {
        console.error('Failed to load stored analysis:', error);
      }
    }
  }, []);

  const generateInsights = (analysisData: any): AnalysisInsight[] => {
    const insights: AnalysisInsight[] = [];
    
    if (analysisData.confidence !== undefined) {
      const confidence = analysisData.confidence;
      if (confidence > 0.8) {
        insights.push({
          category: 'Confidence',
          insight: `High confidence detection (${(confidence * 100).toFixed(1)}%). The models are very certain about this prediction.`,
          confidence: confidence,
          recommendation: 'This result can be trusted with high reliability.'
        });
      } else if (confidence < 0.6) {
        insights.push({
          category: 'Confidence',
          insight: `Low confidence detection (${(confidence * 100).toFixed(1)}%). The models show uncertainty.`,
          confidence: confidence,
          recommendation: 'Consider additional verification or manual review.'
        });
      }
    }

    if (analysisData.analysis?.model_predictions) {
      const models = analysisData.analysis.model_predictions;
      const modelCount = Object.keys(models).length;
      const agreement = Object.values(models).filter((conf: any) => 
        (analysisData.prediction === 'fake' && conf > 0.5) || 
        (analysisData.prediction === 'real' && conf < 0.5)
      ).length;
      
      insights.push({
        category: 'Model Agreement',
        insight: `${agreement}/${modelCount} specialist models agree with the final prediction.`,
        confidence: agreement / modelCount,
        recommendation: agreement === modelCount ? 'Strong consensus among models.' : 'Mixed signals from different models.'
      });
    }

    if (analysisData.analysis?.routing?.video_characteristics) {
      const chars = analysisData.analysis.routing.video_characteristics;
      if (chars.blur_score > 80) {
        insights.push({
          category: 'Video Quality',
          insight: `High blur detected (score: ${chars.blur_score}). This may affect detection accuracy.`,
          confidence: 0.7,
          recommendation: 'Blurry videos are harder to analyze accurately.'
        });
      }
      
      if (chars.brightness < 50) {
        insights.push({
          category: 'Video Quality',
          insight: `Low brightness detected (${chars.brightness}). Dark videos can impact model performance.`,
          confidence: 0.6,
          recommendation: 'Consider enhancing brightness for better analysis.'
        });
      }
    }

    return insights;
  };

  const analyzeJsonData = (data: any) => {
    try {
      const analysisData = typeof data === 'string' ? JSON.parse(data) : data;
      setCurrentAnalysis(analysisData);
      
      const newInsights = generateInsights(analysisData);
      setInsights(newInsights);

      const analysisMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: generateAnalysisResponse(analysisData, newInsights),
        timestamp: new Date(),
        analysisData: analysisData
      };

      setMessages(prev => [...prev, analysisMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: "I couldn't parse that JSON data. Please make sure it's valid JSON from a video analysis.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const generateAnalysisResponse = (data: any, insights: AnalysisInsight[]): string => {
    let response = `## Analysis Summary\n\n`;
    
    response += `**Prediction:** ${data.prediction?.toUpperCase() || 'Unknown'}\n`;
    response += `**Confidence:** ${data.confidence ? (data.confidence * 100).toFixed(1) + '%' : 'N/A'}\n`;
    response += `**File:** ${data.filename || 'Unknown'}\n`;
    response += `**Processing Time:** ${data.processing_time ? data.processing_time.toFixed(2) + 's' : 'N/A'}\n\n`;

    if (data.models_used?.length) {
      response += `**Models Used:** ${data.models_used.join(', ')}\n\n`;
    }

    response += `## Key Insights\n\n`;
    insights.forEach((insight, index) => {
      response += `**${insight.category}:** ${insight.insight}\n`;
      if (insight.recommendation) {
        response += `*Recommendation: ${insight.recommendation}*\n\n`;
      }
    });

    if (data.analysis?.suspicious_frames) {
      response += `**Suspicious Frames:** ${data.analysis.suspicious_frames} frames flagged for review\n\n`;
    }

    response += `Feel free to ask me specific questions about this analysis!`;
    
    return response;
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === 'application/json') {
        // Handle JSON analysis file
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string;
          analyzeJsonData(content);
        };
        reader.readAsText(file);
      } else if (file.type.startsWith('video/')) {
        // Handle video file for media analysis
        analyzeMediaFile(file);
      } else {
        const errorMessage: Message = {
          id: Date.now().toString(),
          type: 'assistant',
          content: "Please upload either a JSON analysis file or a video file (MP4, AVI, MOV, WebM).",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    }
  };

  const analyzeMediaFile = async (file: File) => {
    setIsAnalyzingMedia(true);
    
    const analysisMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: `Analyzing video file "${file.name}" using multiple media analysis APIs. This may take a moment...`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, analysisMessage]);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('analysisType', 'comprehensive');

      const response = await fetch('/api/media-analysis', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const mediaData = await response.json();
        setMediaAnalysis(mediaData);
        
        const resultMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: generateMediaAnalysisResponse(mediaData),
          timestamp: new Date()
        };
        setMessages(prev => [...prev, resultMessage]);
      } else {
        throw new Error('Media analysis failed');
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I couldn't analyze the video file. The media analysis APIs might be unavailable. Try uploading a JSON analysis file instead.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAnalyzingMedia(false);
    }
  };

  const generateMediaAnalysisResponse = (mediaData: any): string => {
    let response = `## Media Analysis Complete\n\n`;
    
    response += `**File:** ${mediaData.filename}\n`;
    response += `**Size:** ${(mediaData.fileSize / (1024 * 1024)).toFixed(2)} MB\n`;
    response += `**APIs Used:** ${mediaData.apisUsed?.join(', ') || 'Internal analysis'}\n\n`;

    if (mediaData.mediaAnalysis?.dimensions) {
      response += `**Video Specs:**\n`;
      response += `- Resolution: ${mediaData.mediaAnalysis.dimensions.width}x${mediaData.mediaAnalysis.dimensions.height}\n`;
      response += `- Duration: ${mediaData.mediaAnalysis.duration?.toFixed(2) || 'Unknown'} seconds\n`;
      if (mediaData.mediaAnalysis.videoSpecs) {
        response += `- Format: ${mediaData.mediaAnalysis.videoSpecs.format}\n`;
        response += `- Frame Rate: ${mediaData.mediaAnalysis.videoSpecs.frameRate} fps\n`;
      }
      response += `\n`;
    }

    if (mediaData.mediaAnalysis?.qualityMetrics) {
      response += `**Quality Analysis:**\n`;
      response += `- Overall Quality Score: ${mediaData.mediaAnalysis.internalAnalysis?.qualityScore?.toFixed(1) || 'N/A'}\n`;
      response += `- Compression Artifacts: ${mediaData.mediaAnalysis.internalAnalysis?.compressionArtifacts ? 'Detected' : 'Not detected'}\n`;
      response += `- Temporal Inconsistencies: ${mediaData.mediaAnalysis.internalAnalysis?.temporalInconsistencies ? 'Found' : 'None found'}\n\n`;
    }

    if (mediaData.mediaAnalysis?.contentTags?.length > 0) {
      response += `**Content Tags:** ${mediaData.mediaAnalysis.contentTags.join(', ')}\n\n`;
    }

    if (mediaData.mediaAnalysis?.audioAnalysis) {
      response += `**Audio Analysis:** Content safety and sentiment analysis enabled\n\n`;
    }

    response += `This comprehensive media analysis can help inform deepfake detection. Ask me specific questions about the video characteristics!`;
    
    return response;
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // Check if user is pasting JSON data
    try {
      const jsonData = JSON.parse(inputMessage);
      if (jsonData.prediction || jsonData.analysis) {
        analyzeJsonData(jsonData);
        setIsLoading(false);
        return;
      }
    } catch {
      // Not JSON, continue with regular message processing
    }

    // Simulate AI response based on current analysis and user question
    try {
      const response = await generateContextualResponse(inputMessage, currentAnalysis);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I'm having trouble processing your request right now. Please try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const generateContextualResponse = async (question: string, analysis: any): Promise<string> => {
    // Try to use the real chat API
    try {
      const response = await fetch('/api/chat-assistant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: question,
          analysisData: analysis,
          conversationHistory: messages.slice(-10).map(m => ({
            role: m.type === 'user' ? 'user' : 'assistant',
            content: m.content
          }))
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.response;
      } else {
        console.warn('Chat API failed, using fallback');
        return generateFallbackResponse(question, analysis);
      }
    } catch (error) {
      console.warn('Chat API error, using fallback:', error);
      return generateFallbackResponse(question, analysis);
    }
  };

  const generateFallbackResponse = (question: string, analysis: any): string => {
    const lowerQuestion = question.toLowerCase();
    
    if (!analysis) {
      return "I don't have any analysis data to work with yet. Please upload a JSON analysis file or paste your analysis results first!";
    }

    if (lowerQuestion.includes('confidence') || lowerQuestion.includes('sure')) {
      const conf = analysis.confidence * 100;
      return `The confidence level is ${conf.toFixed(1)}%. ${conf > 80 ? 'This is quite high and indicates the models are very certain.' : conf < 60 ? 'This is relatively low, suggesting some uncertainty in the prediction.' : 'This is moderate confidence.'}`;
    }

    if (lowerQuestion.includes('model') || lowerQuestion.includes('algorithm')) {
      const models = analysis.models_used || [];
      return `This analysis used ${models.length} specialist models: ${models.join(', ')}. Each model focuses on different aspects like compression artifacts, temporal inconsistencies, and visual anomalies.`;
    }

    if (lowerQuestion.includes('fake') || lowerQuestion.includes('real')) {
      const prediction = analysis.prediction;
      const reasons: string[] = [];
      
      if (analysis.analysis?.model_predictions) {
        const models = analysis.analysis.model_predictions;
        Object.entries(models).forEach(([model, conf]: [string, any]) => {
          if ((prediction === 'fake' && conf > 0.6) || (prediction === 'real' && conf < 0.4)) {
            reasons.push(`${model} model strongly supports this prediction`);
          }
        });
      }
      
      return `The video was classified as ${prediction.toUpperCase()}. ${reasons.length ? 'Key factors: ' + reasons.join(', ') + '.' : 'The models detected patterns consistent with this classification.'}`;
    }

    if (lowerQuestion.includes('quality') || lowerQuestion.includes('blur') || lowerQuestion.includes('brightness')) {
      const chars = analysis.analysis?.routing?.video_characteristics;
      if (chars) {
        return `Video quality analysis: Brightness: ${chars.brightness}, Blur score: ${chars.blur_score}, Resolution: ${chars.width}x${chars.height}. ${chars.blur_score > 80 ? 'High blur detected which may affect accuracy.' : 'Video quality appears acceptable for analysis.'}`;
      }
    }

    if (lowerQuestion.includes('time') || lowerQuestion.includes('process')) {
      return `Processing took ${analysis.processing_time?.toFixed(2) || 'unknown'} seconds. The analysis examined ${analysis.analysis?.frames_analyzed || 'multiple'} frames and generated ${analysis.analysis?.heatmaps_generated || 0} heatmaps for detailed inspection.`;
    }

    // Default response
    return "I can help explain various aspects of your analysis. Try asking about confidence levels, models used, video quality, processing details, or why the video was classified as fake/real.";
  };

  const clearChat = () => {
    setMessages([{
      id: '1',
      type: 'assistant',
      content: "Chat cleared! Upload a new analysis or ask me anything about deepfake detection.",
      timestamp: new Date()
    }]);
    setCurrentAnalysis(null);
    setInsights([]);
  };

  const exportChat = () => {
    const chatData = {
      messages,
      analysis: currentAnalysis,
      insights,
      exportedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-assistant-chat-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              AI Assistant
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Get detailed insights and explanations about your video analysis results
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Chat Interface */}
          <div className="lg:col-span-3">
            <Card className="h-[600px] flex flex-col">
              <CardHeader className="flex-shrink-0 border-b">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <MessageCircle className="w-5 h-5" />
                    Chat with AI Assistant
                  </CardTitle>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={exportChat}>
                      <Download className="w-4 h-4 mr-1" />
                      Export
                    </Button>
                    <Button variant="outline" size="sm" onClick={clearChat}>
                      <Trash2 className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="flex-1 flex flex-col p-0">
                <ScrollArea className="flex-1 p-4">
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-lg p-3 ${
                            message.type === 'user'
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
                          }`}
                        >
                          <div className="whitespace-pre-wrap text-sm">
                            {message.content}
                          </div>
                          <div className="text-xs opacity-70 mt-1">
                            {message.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-3">
                          <div className="flex items-center gap-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              Analyzing...
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  <div ref={messagesEndRef} />
                </ScrollArea>

                <div className="border-t p-4">
                  <div className="flex gap-2 mb-2">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".json,video/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isAnalyzingMedia}
                    >
                      <Upload className="w-4 h-4 mr-1" />
                      {isAnalyzingMedia ? 'Analyzing...' : 'Upload File'}
                    </Button>
                  </div>
                  
                  <div className="flex gap-2">
                    <Textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      placeholder="Ask about your analysis, upload JSON/video files, or paste JSON data..."
                      className="flex-1 min-h-[60px]"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                    />
                    <Button 
                      onClick={handleSendMessage}
                      disabled={!inputMessage.trim() || isLoading || isAnalyzingMedia}
                      className="self-end"
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Insights Panel */}
          <div className="space-y-6">
            {currentAnalysis && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <FileText className="w-4 h-4" />
                    Current Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <Badge variant={currentAnalysis.prediction === 'fake' ? 'destructive' : 'default'}>
                      {currentAnalysis.prediction?.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="text-sm">
                    <div className="font-medium">Confidence</div>
                    <div className="text-gray-600 dark:text-gray-400">
                      {(currentAnalysis.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-sm">
                    <div className="font-medium">File</div>
                    <div className="text-gray-600 dark:text-gray-400 truncate">
                      {currentAnalysis.filename}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {insights.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Zap className="w-4 h-4" />
                    Key Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {insights.map((insight, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Badge variant="outline" className="text-xs">
                          {insight.category}
                        </Badge>
                        <div className="text-xs text-gray-500">
                          {(insight.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-sm text-gray-700 dark:text-gray-300">
                        {insight.insight}
                      </div>
                      {insight.recommendation && (
                        <div className="text-xs text-blue-600 dark:text-blue-400 italic">
                          {insight.recommendation}
                        </div>
                      )}
                      {index < insights.length - 1 && <Separator />}
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setInputMessage("Explain the confidence level")}
                >
                  Explain Confidence
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setInputMessage("Which models were used?")}
                >
                  Model Details
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setInputMessage("Analyze video quality")}
                >
                  Video Quality
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start"
                  onClick={() => setInputMessage("Why was this classified as fake/real?")}
                >
                  Classification Reason
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;