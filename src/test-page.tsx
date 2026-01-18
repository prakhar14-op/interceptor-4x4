import React from 'react';

export default function TestPage() {
  return (
    <div style={{ padding: '50px', textAlign: 'center', fontFamily: 'Arial' }}>
      <h1>🎉 React App is Working!</h1>
      <p>If you can see this, your React app is loading correctly.</p>
      <button 
        onClick={() => alert('JavaScript is working!')}
        style={{
          background: '#007bff',
          color: 'white',
          border: 'none',
          padding: '10px 20px',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Test JavaScript
      </button>
    </div>
  );
}