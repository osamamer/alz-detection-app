import React, { useRef, useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import CanvasDraw from 'react-canvas-draw';

export const ClockDrawingTab = () => {
    const canvasRef = useRef<CanvasDraw | null>(null);
    const [submitted, setSubmitted] = useState(false);
    const [result, setResult] = useState<string | null>(null);

    const handleSubmit = () => {
        if (canvasRef.current) {
            const drawingData = canvasRef.current.getSaveData(); // Get drawing data
            setSubmitted(true);

            // Send the drawing data to the backend
            fetch('http://127.0.0.1:5000/analyze-clock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ drawingData }),
            })
                .then((response) => response.json())
                .then((data) => {
                    setResult(data.result || 'No insights available.');
                    setSubmitted(false);
                })
                .catch((error) => {
                    console.error('Error:', error);
                    setSubmitted(false);
                });
        }
    };

    return (
        <Box textAlign="center">
            <Typography variant="h4" gutterBottom>
                Clock Drawing Test
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Please draw a clock showing the time "10:10" in the canvas below.
            </Typography>

            <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
                <CanvasDraw
                    ref={canvasRef}
                    canvasWidth={400}
                    canvasHeight={400}
                    brushRadius={2}
                    lazyRadius={2}
                    style={{ border: '1px solid #ccc' }}
                />
            </Box>

            <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                disabled={submitted}
                sx={{ px: 4, py: 2 }}
            >
                {submitted ? 'Submitting...' : 'Submit Drawing'}
            </Button>

            {result && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6" gutterBottom>
                        Analysis Result:
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        {result}
                    </Typography>
                </Box>
            )}
        </Box>
    );
};

export default ClockDrawingTab;
