import React, { useEffect, useRef, useState } from 'react';
import { PuzzleState } from '@shared/types/puzzle';
import { ImageProcessor } from '@shared/utils/imageProcessor';

interface CameraProps {
  onPuzzleDetected: (state: PuzzleState) => void;
  isARMode: boolean;
}

export function Camera({ onPuzzleDetected, isARMode }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const imageProcessor = useRef<ImageProcessor | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async (facingMode: 'environment' | 'user' = 'environment') => {
    console.log('Starting camera with facing mode:', facingMode);
    try {
      // Stop any existing stream
      if (streamRef.current) {
        console.log('Stopping existing stream...');
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // Ensure we have a valid ImageProcessor instance
      if (!imageProcessor.current) {
        console.log('Reinitializing ImageProcessor...');
        imageProcessor.current = await ImageProcessor.getInstance();
      }

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: facingMode
        }
      };

      console.log('Requesting camera access with constraints:', constraints);
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        console.log('Setting video source...');
        videoRef.current.srcObject = stream;
      }

      setIsStreaming(true);
      setIsInitialized(true);
      console.log('Camera started successfully');

      // Start processing frames
      const processFrame = async () => {
        if (!videoRef.current || !imageProcessor.current || !isInitialized) {
          console.log('Skipping frame processing:', {
            hasVideo: !!videoRef.current,
            hasProcessor: !!imageProcessor.current,
            isInitialized
          });
          return;
        }

        try {
          // Only process frames in AR mode
          if (isARMode) {
            console.log('Processing frame in AR mode...');
            const puzzleState = await imageProcessor.current.detectPuzzleInAR(videoRef.current);
            if (puzzleState && puzzleState.size.rows > 1 && puzzleState.size.cols > 1) {
              console.log('AR detection result:', puzzleState);
              onPuzzleDetected(puzzleState);
            }
          }
          // In normal mode, we don't process frames continuously
        } catch (err) {
          console.error('Error processing frame:', err);
        }

        requestAnimationFrame(processFrame);
      };

      console.log('Starting frame processing loop...');
      processFrame();
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Failed to access camera. Please ensure you have granted camera permissions.');
    }
  };

  useEffect(() => {
    console.log('Camera component mounted');
    const initialize = async () => {
      try {
        console.log('Starting initialization...');
        // Initialize image processor
        console.log('Getting ImageProcessor instance...');
        imageProcessor.current = await ImageProcessor.getInstance();
        
        console.log('Waiting for OpenCV to load...');
        await new Promise(resolve => setTimeout(resolve, 1000)); // Give OpenCV time to load

        // Start with back camera
        console.log('Starting camera...');
        await startCamera('environment');
      } catch (err) {
        console.error('Failed to initialize:', err);
        setError('Failed to initialize camera and image processing. Please refresh the page.');
      }
    };

    initialize();

    return () => {
      console.log('Camera component unmounting');
      if (streamRef.current) {
        console.log('Stopping camera stream...');
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      // Clean up ImageProcessor when component unmounts
      ImageProcessor.cleanup();
    };
  }, []);

  useEffect(() => {
    if (isARMode && isStreaming) {
      console.log('AR mode enabled, switching to back camera...');
      startCamera('environment');
    }
  }, [isARMode]);

  const captureImage = async () => {
    console.log('Capture button clicked');
    if (!videoRef.current || !imageProcessor.current || isCapturing) {
      return;
    }
    
    setIsCapturing(true);
    setError(null);
    
    try {
      console.log('Starting image capture...');
      // Create a canvas to capture the current frame
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      console.log('Canvas created with dimensions:', canvas.width, 'x', canvas.height);
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }

      console.log('Drawing image to canvas...');
      ctx.drawImage(videoRef.current, 0, 0);
      
      // Create and load image
      console.log('Creating image from canvas...');
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = canvas.toDataURL('image/png');
      });
      
      console.log('Image loaded, dimensions:', image.width, 'x', image.height);
      console.log('Processing image...');
      
      const puzzleState = await imageProcessor.current.processImage(image);
      console.log('Image processed:', puzzleState);
      
      // Check if the puzzle state is valid (non-empty grid)
      if (puzzleState.size.rows <= 1 && puzzleState.size.cols <= 1) {
        console.log('No valid puzzle detected in image');
        setError('No puzzle detected in image. Please try again with a clear view of a sliding puzzle.');
        return;
      }
      
      console.log('Puzzle detected, calling onPuzzleDetected...');
      onPuzzleDetected(puzzleState);
    } catch (error) {
      console.error('Error capturing image:', error);
      setError(`Failed to capture and process image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsCapturing(false);
      console.log('Capture process completed');
    }
  };

  const switchCamera = () => {
    console.log('Switch camera button clicked');
    const currentFacingMode = streamRef.current?.getVideoTracks()[0].getSettings().facingMode;
    console.log('Current facing mode:', currentFacingMode);
    startCamera(currentFacingMode === 'environment' ? 'user' : 'environment');
  };

  const clearError = () => {
    console.log('Clearing error and returning to capture mode');
    
    // Reset error state
    setError(null);
    
    // Restart camera if it's not streaming
    if (!isStreaming || !streamRef.current) {
      console.log('Camera is not streaming, restarting camera');
      startCamera('environment'); // Force restart with back camera
    } else {
      console.log('Camera is already streaming');
    }
  };

  return (
    <div className="relative w-full h-full">
      {error ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-100 p-4">
          <p className="text-red-500 text-center mb-4">{error}</p>
          <button
            onClick={clearError}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Try Again
          </button>
        </div>
      ) : (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
          <div className="absolute bottom-4 left-0 right-0 flex justify-center space-x-4">
            <button
              onClick={captureImage}
              disabled={isCapturing || !isInitialized}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                isCapturing || !isInitialized
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-700'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
            >
              {isCapturing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                'Capture Puzzle'
              )}
            </button>
            <button
              onClick={switchCamera}
              disabled={isCapturing || !isInitialized}
              className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                isCapturing || !isInitialized
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-700'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500`}
            >
              Switch Camera
            </button>
          </div>
        </>
      )}
    </div>
  );
} 