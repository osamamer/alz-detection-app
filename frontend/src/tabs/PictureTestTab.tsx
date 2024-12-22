import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Card, CardContent, CardMedia, List, ListItem, ListItemText, Divider } from '@mui/material';

function PictureTestTab() {
    const [description, setDescription] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<any | null>(null); // Update to handle structured response

    const handleSubmit = () => {
        setSubmitted(true);

        fetch('http://127.0.0.1:5000/analyze-picture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description }),
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                setAnalysisResult(data); // Store the structured response
                setSubmitted(false);
            })
            .catch((error) => {
                console.error('Error:', error);
                setAnalysisResult(null); // Handle error gracefully
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
                    image="cookie-theft-pic.png" // Relative to the public folder
                    alt="Woman in the Kitchen"
                    sx={{ height: 400 }}
                />
                <CardContent>
                    <Typography variant="body2" color="text.secondary">
                        Look at the image and describe what you see in as much detail as possible.
                    </Typography>
                </CardContent>
            </Card>

            {/* Input and Button */}
            <Box sx={{ maxWidth: 600, mx: 'auto', mb: 4 }}>
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
            {/* Analysis Result */}
            {/* Analysis Result */}
            {analysisResult && (
                <Box
                    sx={{
                        mt: 4,
                        maxWidth: 600,
                        mx: 'auto',
                        textAlign: 'left',
                        border: '1px solid #ddd',
                        borderRadius: 2,
                        p: 3,
                        // backgroundColor: '#f9f9f9',
                    }}
                >
                    <Typography variant="h6" gutterBottom>
                        Analysis Result
                    </Typography>

                    {/* Dynamic Risk Level Message */}
                    <Typography
                        variant="body1"
                        sx={{
                            fontWeight: 'bold',
                            color:
                                analysisResult.risk_level === 'High'
                                    ? 'error.main'
                                    : analysisResult.risk_level === 'Medium'
                                        ? 'warning.main'
                                        : 'success.main',
                            mb: 2,
                        }}
                    >
                        {analysisResult.risk_level === 'Low' && (
                            <>
                                Your description appears to be sufficient and within normal range.
                            </>
                        )}
                        {analysisResult.risk_level === 'Medium' && (
                            <>
                                Your description raises some moderate concerns. While this is not alarming, it
                                might be a good idea to consult with a professional for a more thorough
                                evaluation.
                            </>
                        )}
                        {analysisResult.risk_level === 'High' && (
                            <>
                                Your description indicates significant concerns. It is strongly recommended
                                that you seek expert advice or a comprehensive cognitive assessment.
                            </>
                        )}
                    </Typography>

                    {/* Concerns */}
                    {analysisResult.concerns.length > 0 && (
                        <Typography variant="body1" sx={{ mb: 2 }}>
                            {analysisResult.concerns[0] === 'No significant concerns identified' ? (
                                <>No significant concerns were identified.</>
                            ) : (
                                <>Some areas of concern include: {analysisResult.concerns.join(', ')}.</>
                            )}
                        </Typography>
                    )}

                    {/* Scoring */}
                    <Typography variant="body1" sx={{ mt: 3 }}>
                        {analysisResult.risk_level === 'Low' ? (
                            <>If you&#39;d like a closer look at how you scored:</>
                        ) : (
                            <>Here are the detailed scoring insights:</>
                        )}
                    </Typography>
                    <Typography variant="body1">
                        Cognitive Markers: {analysisResult.scoring.cognitive_markers}, Information
                        Completeness: {analysisResult.scoring.information_completeness}, and Linguistic
                        Complexity: {analysisResult.scoring.linguistic_complexity}.
                    </Typography>
                </Box>
            )}

        </Box>
    );
}

export default PictureTestTab;
