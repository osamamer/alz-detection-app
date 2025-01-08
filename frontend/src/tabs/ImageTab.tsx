import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Card, CardContent, CardMedia, CircularProgress, Alert } from '@mui/material';

interface PredictionResult {
    result: {
        predicted_class: string;
        probabilities: {
            [key: string]: number;
        };
    };
}

const ImageTab = () => {
    const [imageType, setImageType] = useState<'2d' | '3d' | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        try {
            setIsLoading(true);
            setError(null);
            setResult(null);

            if (event.target.files && event.target.files[0]) {
                const file = event.target.files[0];
                const formData = new FormData();
                formData.append('image', file);

                if (imageType === '2d') {
                    // Handle 2D image preview
                    const reader = new FileReader();
                    reader.onload = () => setImagePreview(reader.result as string);
                    reader.readAsDataURL(file);

                    // Send to backend
                    const response = await fetch('/predict-2d-image', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    setResult(data);
                } else if (imageType === '3d') {
                    if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
                        throw new Error('Please upload a valid NIfTI file (.nii or .nii.gz)');
                    }

                    const response = await fetch('http://127.0.0.1:5000/predict-3d-image', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    setResult(data);
                }
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred during upload');
            console.error('Upload error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const getRecommendation = (predictedClass: string) => {
        switch (predictedClass) {
            case 'CN':
                return "Based on the analysis, your brain scan appears normal. Continue maintaining a healthy lifestyle with regular exercise and cognitive activities.";
            case 'MCI':
                return "The scan indicates mild cognitive impairment. We recommend scheduling a follow-up with a neurologist for a comprehensive evaluation and discussing potential interventions.";
            case 'AD':
                return "The scan suggests patterns consistent with Alzheimer's Disease. Please consult with a neurologist as soon as possible to discuss treatment options and support services.";
            default:
                return "Please consult with your healthcare provider to discuss these results in detail.";
        }
    };

    const renderPredictionResult = (result: PredictionResult) => {
        const confidenceLevel = result.result.probabilities[result.result.predicted_class] * 100;

        return (
            <Card sx={{ mt: 4, maxWidth: 800, mx: 'auto', p: 3 }}>
                <Typography variant="h5" gutterBottom color="primary">
                    Analysis Results
                </Typography>

                <Typography variant="body1" sx={{ mt: 2, mb: 3 }}>
                    Dear Patient,
                </Typography>

                <Typography variant="body1" paragraph>
                    We have analyzed your brain scan using our advanced AI system.
                    The analysis was completed with {confidenceLevel.toFixed(1)}% confidence.
                </Typography>

                <Typography variant="body1" paragraph>
                    {getRecommendation(result.result.predicted_class)}
                </Typography>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                        Detailed Analysis:
                    </Typography>
                    {Object.entries(result.result.probabilities).map(([className, probability]) => (
                        <Typography key={className} variant="body2" sx={{ my: 1 }}>
                            {className}: {(probability * 100).toFixed(1)}%
                        </Typography>
                    ))}
                </Box>

                <Typography variant="body2" sx={{ mt: 3, color: 'text.secondary', fontStyle: 'italic' }}>
                    Note: This is an AI-assisted analysis and should not be considered as a final diagnosis.
                    Please consult with a qualified healthcare professional for proper medical evaluation and diagnosis.
                </Typography>
            </Card>
        );
    };

    return (
        <Box textAlign="center">
            <Typography variant="h4" gutterBottom>
                Image Processing
            </Typography>

            <Box>
                <Button
                    variant="contained"
                    sx={{ mx: 2, px: 4, py: 2, fontSize: '1rem' }}
                    onClick={() => {
                        setImageType('2d');
                        setResult(null);
                        setError(null);
                        setImagePreview(null);
                    }}
                >
                    2D Image
                </Button>
                <Button
                    variant="contained"
                    sx={{ mx: 2, px: 4, py: 2, fontSize: '1rem' }}
                    onClick={() => {
                        setImageType('3d');
                        setResult(null);
                        setError(null);
                        setImagePreview(null);
                    }}
                >
                    3D Image
                </Button>
            </Box>

            {imageType && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6">
                        Upload {imageType === '2d' ? '2D' : '3D'} Image
                    </Typography>
                    <TextField
                        type="file"
                        inputProps={{
                            accept: imageType === '2d' ? 'image/*' : '.nii,.nii.gz',
                        }}
                        onChange={handleImageUpload}
                        fullWidth
                        sx={{ mt: 2 }}
                        disabled={isLoading}
                    />

                    {isLoading && (
                        <Box sx={{ mt: 2 }}>
                            <CircularProgress />
                            <Typography>Processing image...</Typography>
                        </Box>
                    )}

                    {error && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                            {error}
                        </Alert>
                    )}

                    {result && renderPredictionResult(result)}

                    {imageType === '2d' && imagePreview && (
                        <Card sx={{ mt: 4, maxWidth: 600, mx: 'auto' }}>
                            <CardMedia
                                component="img"
                                alt="Image Preview"
                                height="300"
                                image={imagePreview}
                            />
                            <CardContent>
                                <Typography variant="body2" color="text.secondary">
                                    Preview of the uploaded 2D image.
                                </Typography>
                            </CardContent>
                        </Card>
                    )}
                </Box>
            )}
        </Box>
    );
};

export default ImageTab;