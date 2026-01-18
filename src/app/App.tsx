import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { ArchitectureProvider } from './context/ArchitectureContext';
import Navbar from './components/Navbar';

// Import working pages
import Home from './pages/Home';
import AnalysisWorkbench from './pages/AnalysisWorkbench';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import AdvancedAnalytics from './pages/AdvancedAnalytics';
import AIAssistant from './pages/AIAssistant';
import Contact from './pages/Contact';
import FAQ from './pages/FAQ';
import ChartDemo from './pages/ChartDemo';

export default function App() {
  return (
    <ThemeProvider>
      <ArchitectureProvider>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-purple-100 via-blue-100 to-cyan-100 dark:from-gray-950 dark:via-blue-950 dark:to-purple-950 transition-colors relative overflow-hidden">
            {/* Simple gradient background instead of complex LiquidEther */}
            <div 
              className="fixed inset-0 pointer-events-none opacity-20 dark:opacity-30" 
              style={{ 
                zIndex: 1,
                background: 'linear-gradient(45deg, #3B82F6, #A78BFA, #60A5FA)',
                filter: 'blur(100px)'
              }}
            />
            
            <div className="relative z-10">
              <Navbar />
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/workbench" element={<AnalysisWorkbench />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/advanced-analytics" element={<AdvancedAnalytics />} />
                <Route path="/assistant" element={<AIAssistant />} />
                <Route path="/contact" element={<Contact />} />
                <Route path="/faq" element={<FAQ />} />
                <Route path="/chart-demo" element={<ChartDemo />} />
              </Routes>
            </div>
          </div>
        </Router>
      </ArchitectureProvider>
    </ThemeProvider>
  );
}
