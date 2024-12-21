import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Card, CardContent, CardMedia } from '@mui/material';

const ImageTab = () => {
    const [imageType, setImageType] = useState<'2d' | '3d' | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            if (imageType === '2d') {
                const reader = new FileReader();
                reader.onload = () => setImagePreview(reader.result as string);
                reader.readAsDataURL(file);

                const formData = new FormData();
                formData.append('image', file);
                fetch('/predict-2d-image', { method: 'POST', body: formData });
            } else if (imageType === '3d') {
                const formData = new FormData();
                formData.append('image', file);
                fetch('/predict-3d-image', { method: 'POST', body: formData });
            }
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
                    onClick={() => setImageType('2d')}
                >
                    2D Image
                </Button>
                <Button
                    variant="contained"
                    sx={{ mx: 2, px: 4, py: 2, fontSize: '1rem' }}
                    onClick={() => setImageType('3d')}
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
                    />
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
