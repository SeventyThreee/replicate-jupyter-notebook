
import React, { useState, useEffect } from 'react';

const JupyterHeader: React.FC = () => {
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(prev => {
        // Reset to 0 after reaching 59 minutes (3540 seconds)
        return prev >= 3540 ? 0 : prev + 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatElapsedTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds} second${seconds !== 1 ? 's' : ''} ago`;
    } else {
      const minutes = Math.floor(seconds / 60);
      return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    }
  };

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-jupyter-border">
      <div className="flex items-center space-x-2">
        <div className="flex items-center">
          {/* Updated Jupyter logo using the uploaded image */}
          <img 
            src="/lovable-uploads/d3461be3-1cd8-4c6e-a33e-5a52b341c046.png" 
            alt="Jupyter Logo" 
            className="h-10 w-auto"
          />
          <span className="ml-2 text-xl">exam</span>
          <span className="ml-2 text-gray-600">Last Checkpoint: {formatElapsedTime(elapsedTime)}</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex items-center">
          {/* Python logo */}
          <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
            <linearGradient id="goQNIBi_YLQdp9k2YC4bva" x1="10.458" x2="26.314" y1="12.492" y2="26.192" gradientTransform="matrix(1 0 0 -1 0 48)" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#26abe7"/>
              <stop offset="1" stopColor="#086dbf"/>
            </linearGradient>
            <path fill="url(#goQNIBi_YLQdp9k2YC4bva)" d="M24.047,5c-1.555,0.005-2.633,0.142-3.936,0.367c-3.848,0.67-4.549,2.077-4.549,4.67V14h9v2 H15.22h-4.35c-2.636,0-4.943,1.242-5.674,4.219c-0.826,3.417-0.863,5.557,0,9.125C5.851,32.005,7.294,34,9.931,34h3.632v-5.104 c0-2.966,2.686-5.896,5.764-5.896h7.236c2.523,0,5-1.862,5-4.377v-8.586c0-2.439-1.759-4.263-4.218-4.672 C27.406,5.359,25.589,4.994,24.047,5z M19.063,9c0.821,0,1.5,0.677,1.5,1.502c0,0.833-0.679,1.498-1.5,1.498 c-0.837,0-1.5-0.664-1.5-1.498C17.563,9.68,18.226,9,19.063,9z"/>
            <linearGradient id="goQNIBi_YLQdp9k2YC4bvb" x1="35.334" x2="23.517" y1="37.911" y2="21.034" gradientTransform="matrix(1 0 0 -1 0 48)" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#feb705"/>
              <stop offset="1" stopColor="#ffda1c"/>
            </linearGradient>
            <path fill="url(#goQNIBi_YLQdp9k2YC4bvb)" d="M23.078,43c1.555-0.005,2.633-0.142,3.936-0.367c3.848-0.67,4.549-2.077,4.549-4.67V34h-9v-2 h9.343h4.35c2.636,0,4.943-1.242,5.674-4.219c0.826-3.417,0.863-5.557,0-9.125C41.274,15.995,39.831,14,37.194,14h-3.632v5.104 c0,2.966-2.686,5.896-5.764,5.896h-7.236c-2.523,0-5,1.862-5,4.377v8.586c0,2.439,1.759,4.263,4.218,4.672 C19.719,42.641,21.536,43.006,23.078,43z M28.063,39c-0.821,0-1.5-0.677-1.5-1.502c0-0.833,0.679-1.498,1.5-1.498 c0.837,0,1.5,0.664,1.5,1.498C29.563,38.32,28.899,39,28.063,39z"/>
          </svg>
          <button className="ml-2 px-3 py-1 bg-white border border-gray-300 rounded-sm hover:bg-gray-100 transition-colors duration-200">
            Quit
          </button>
        </div>
      </div>
    </div>
  );
};

export default JupyterHeader;
