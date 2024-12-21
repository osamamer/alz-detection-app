import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Card, CardContent, CardMedia } from '@mui/material';

function PictureTestTab() {
    const [description, setDescription] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<string | null>(null);

    const handleSubmit = () => {
        setSubmitted(true);

        // Send description to backend
        fetch('http://127.0.0.1:5000/analyze-picture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description }),
        })
            .then((response) => response.json())
            .then((data) => {
                setAnalysisResult(data.result || 'No insights available.');
                setSubmitted(false);
            })
            .catch((error) => {
                console.error('Error:', error);
                setSubmitted(false);
            });
    };

    return (
        <Box sx={{ textAlign: 'center', py: 6 }}>
            {/* Title */}
            <Typography variant="h4" gutterBottom>
                Picture Test
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Describe what you observe in the picture below. Your response will help assess your memory and interpretation skills.
            </Typography>

            {/* Image Display */}
            <Card sx={{ maxWidth: 600, mx: 'auto', mb: 4 }}>
                <CardMedia
                    component="img"
                    image="/images/woman_in_kitchen.jpg" // Replace with your image path
                    alt="Woman in the Kitchen"
                    sx={{ height: 400 }}
                />
                <CardContent>
                    <Typography variant="body2" color="text.secondary">
                        Look at the image and describe what you see in as much detail as possible.
                    </Typography>
                </CardContent>
            </Card>

            {/* Input Field */}
            <Box sx={{ maxWidth: 600, mx: 'auto' }}>
                <TextField
                    label="Your Description"
                    placeholder="Write your observations here..."
                    multiline
                    fullWidth
                    rows={6}
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    sx={{ mb: 3 }}
                />
                <Button
                    variant="contained"
                    color="primary"
                    disabled={submitted || !description.trim()}
                    onClick={handleSubmit}
                    sx={{ px: 4, py: 2, fontSize: '1rem' }}
                >
                    {submitted ? 'Submitting...' : 'Submit'}
                </Button>
            </Box>

            {/* Analysis Result */}
            {analysisResult && (
                <Box sx={{ mt: 4, maxWidth: 600, mx: 'auto' }}>
                    <Typography variant="h6" gutterBottom>
                        Analysis Result:
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        {analysisResult}
                    </Typography>
                </Box>
            )}
        </Box>
    );
}

export default PictureTestTab;
