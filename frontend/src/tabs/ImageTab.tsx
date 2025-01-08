import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Card, CardContent, CardMedia, CircularProgress, Alert } from '@mui/material';

const ImageTab = () => {
    const [imageType, setImageType] = useState<'2d' | '3d' | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<string | null>(null);
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
                    setResult(data.result);

                } else if (imageType === '3d') {
                    // Verify file type for 3D
                    if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
                        throw new Error('Please upload a valid NIfTI file (.nii or .nii.gz)');
                    }

                    // Send to backend
                    const response = await fetch('http://127.0.0.1:5000//predict-3d-image', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    setResult(data.result);
                }
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred during upload');
            console.error('Upload error:', err);
        } finally {
            setIsLoading(false);
        }
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

                    {result && (
                        <Alert severity="success" sx={{ mt: 2 }}>
                            Prediction Result: {result}
                        </Alert>
                    )}

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