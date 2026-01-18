import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '50px', 
          textAlign: 'center', 
          fontFamily: 'Arial',
          background: '#f8f9fa',
          minHeight: '100vh'
        }}>
          <h1 style={{ color: '#dc3545' }}>🚨 Something went wrong!</h1>
          <h2>React Error Detected</h2>
          <div style={{ 
            background: '#fff', 
            padding: '20px', 
            margin: '20px auto', 
            maxWidth: '600px',
            border: '1px solid #dee2e6',
            borderRadius: '8px',
            textAlign: 'left'
          }}>
            <h3>Error Details:</h3>
            <pre style={{ 
              background: '#f8f9fa', 
              padding: '10px', 
              borderRadius: '4px',
              overflow: 'auto',
              fontSize: '14px'
            }}>
              {this.state.error?.toString()}
            </pre>
            <h3>Stack Trace:</h3>
            <pre style={{ 
              background: '#f8f9fa', 
              padding: '10px', 
              borderRadius: '4px',
              overflow: 'auto',
              fontSize: '12px',
              maxHeight: '200px'
            }}>
              {this.state.error?.stack}
            </pre>
          </div>
          <button 
            onClick={() => window.location.reload()}
            style={{
              background: '#007bff',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Reload Page
          </button>
          <div style={{ marginTop: '20px', fontSize: '14px', color: '#6c757d' }}>
            <p>Check the browser console (F12) for more details.</p>
            <p>This error boundary will help identify the problematic component.</p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;