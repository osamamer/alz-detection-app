import React, { useState } from 'react';
import { Box, Typography, Button, IconButton, TextField, Card, CardContent, CardMedia, CircularProgress, Alert } from '@mui/material';

interface PredictionResult {
    result: {
        predicted_class: string;
        probabilities: {
            [key: string]: number;
        };
    };
}

const ImageTab = () => {
    const [imageType, setImageType] = useState<'dicom' | '3d' | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);

    const validateFiles = (files: FileList): string | null => {
        if (imageType === 'dicom') {
            // Check if any files were selected
            if (files.length === 0) {
                return 'Please select at least one DICOM file';
            }

            // Check if at least 10 files for a meaningful 3D volume
            if (files.length < 10) {
                return 'Please select more DICOM files. A proper brain MRI series typically contains many slices (usually more than 100)';
            }

            // Check if files are too many (arbitrary limit to prevent memory issues)
            if (files.length > 500) {
                return 'Too many files selected. Please check your selection';
            }

            // Validate each file
            for (let i = 0; i < files.length; i++) {
                const file = files[i];

                // Check file extension
                if (!file.name.toLowerCase().endsWith('.dcm') &&
                    !file.name.toLowerCase().endsWith('.dicom')) {
                    return `File "${file.name}" is not a DICOM file. Please only select DICOM files (.dcm or .dicom)`;
                }

                // Check file size (arbitrary 10MB limit per file)
                if (file.size > 10 * 1024 * 1024) {
                    return `File "${file.name}" is too large. Each file should be less than 10MB`;
                }
            }
        } else if (imageType === '3d') {
            if (files.length !== 1) {
                return 'Please select exactly one NIfTI file';
            }

            const file = files[0];
            if (!file.name.toLowerCase().endsWith('.nii') &&
                !file.name.toLowerCase().endsWith('.nii.gz')) {
                return 'Please upload a valid NIfTI file (.nii or .nii.gz)';
            }

            // Check file size (arbitrary 100MB limit)
            if (file.size > 100 * 1024 * 1024) {
                return 'File is too large. Please ensure your NIfTI file is less than 100MB';
            }
        }

        return null;
    };

    const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        try {
            setError(null);
            setResult(null);

            if (!event.target.files || event.target.files.length === 0) {
                setError('No files selected');
                setSelectedFiles(null);
                return;
            }

            // Set selected files first so the UI updates
            setSelectedFiles(event.target.files);

            // Then validate files
            const validationError = validateFiles(event.target.files);
            if (validationError) {
                setError(validationError);
                event.target.value = ''; // Reset file input
                setSelectedFiles(null);
                return;
            }

            setIsLoading(true);

            const formData = new FormData();

            if (imageType === 'dicom') {
                // For DICOM, append all files with array notation
                Array.from(event.target.files).forEach((file) => {
                    formData.append('dicom_files[]', file);
                });
            } else {
                formData.append('image', event.target.files[0]);
            }

            const response = await fetch('http://127.0.0.1:5000/predict-3d-image', {
                method: 'POST',
                body: formData
            });

            const contentType = response.headers.get('content-type');
            if (!response.ok) {
                let errorMessage = 'An error occurred during upload';
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } else {
                    const textError = await response.text();
                    errorMessage = textError || `HTTP error! status: ${response.status}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            setResult(data);

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

    const getDiagnosisColor = (predictedClass: string): string => {
        switch (predictedClass) {
            case 'CN':
                return '#4caf50'; // green
            case 'MCI':
                return '#ff9800'; // yellow/orange
            case 'AD':
                return '#f44336'; // red
            default:
                return 'inherit';
        }
    };

    const getFullClassName = (abbreviation: string): string => {
        switch (abbreviation) {
            case 'CN':
                return 'Cognitively Normal';
            case 'MCI':
                return 'Mild Cognitive Impairment';
            case 'AD':
                return "Alzheimer's Disease";
            default:
                return abbreviation;
        }
    };

    const renderPredictionResult = (result: PredictionResult) => {
        const confidenceLevel = result.result.probabilities[result.result.predicted_class] * 100;
        const diagnosisColor = getDiagnosisColor(result.result.predicted_class);

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
                    The analysis indicates a pattern consistent with{' '}
                    <Typography component="span" sx={{
                        color: diagnosisColor,
                        fontWeight: 'bold',
                        display: 'inline'
                    }}>
                        {getFullClassName(result.result.predicted_class)}
                    </Typography>{' '}
                    with {confidenceLevel.toFixed(1)}% confidence.
                </Typography>

                <Typography variant="body1" paragraph>
                    {getRecommendation(result.result.predicted_class)}
                </Typography>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                        Model Probabilities:
                    </Typography>
                    {Object.entries(result.result.probabilities).map(([className, probability]) => (
                        <Typography
                            key={className}
                            variant="body2"
                            sx={{
                                my: 1,
                                color: getDiagnosisColor(className)
                            }}
                        >
                            {getFullClassName(className)}: {(probability * 100).toFixed(1)}%
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

    const renderFileUploadInstructions = () => {
        if (imageType === 'dicom') {
            return (
                <Box sx={{ mt: 2, mb: 2 }}>
                    <Typography variant="body1" color="text.secondary">
                        Please select all DICOM files (.dcm) from your MRI scan series.
                        You can select multiple files at once.
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Note: A complete brain MRI typically consists of multiple DICOM files
                        (usually over 100 slices) that together form a 3D image. Each file should be less than 10MB.
                    </Typography>
                </Box>
            );
        } else if (imageType === '3d') {
            return (
                <Box sx={{ mt: 2, mb: 2 }}>
                    <Typography variant="body1" color="text.secondary">
                        Please upload a NIfTI file (.nii or .nii.gz) containing your 3D brain scan.
                        The file should be less than 100MB.
                    </Typography>
                </Box>
            );
        }
        return null;
    };

    return (
        <Box textAlign="center">
            <Typography variant="h4" gutterBottom>
                Brain Scan Analysis
            </Typography>

            <Box>
                <Button
                    variant="contained"
                    startIcon={
                        <Box
                            component="img"
                            src="/2d-brain.png"
                            alt="2d-brain"
                            sx={{
                                height: 28,
                                width: 28,
                            }}
                        />
                    }
                    sx={{
                        mx: 2,
                        px: 4,
                        py: 2,
                        fontSize: '1rem',
                        '& .MuiButton-startIcon': {
                            marginRight: 1.5
                        },
                        width: '30%'
                    }}
                    onClick={() => {
                        setImageType('dicom');
                        setResult(null);
                        setError(null);
                        setImagePreview(null);
                        setSelectedFiles(null);
                    }}
                >
                    DICOM Files
                </Button>
                <Button
                    variant="contained"
                    onClick={() => {
                        setImageType('3d');
                        setResult(null);
                        setError(null);
                        setImagePreview(null);
                        setSelectedFiles(null);
                    }}
                    startIcon={
                        <Box
                            component="img"
                            src="/3d-brain.png"
                            alt="2d-brain"
                            sx={{
                                height: 28,
                                width: 28,
                            }}
                        />
                    }
                    sx={{
                        mx: 2,
                        px: 4,
                        py: 2,
                        fontSize: '1rem',
                        '& .MuiButton-startIcon': {
                            marginRight: 1.5
                        },
                        width: '30%'

                    }}
                >
                    NIfTI File
                </Button>
            </Box>

            {imageType && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6">
                        Upload {imageType === 'dicom' ? 'DICOM Files' : 'NIfTI File'}
                    </Typography>

                    {renderFileUploadInstructions()}

                    <TextField
                        type="file"
                        inputProps={{
                            accept: imageType === 'dicom' ? '.dcm,.dicom' : '.nii,.nii.gz',
                            multiple: imageType === 'dicom'
                        }}
                        onChange={handleImageUpload}
                        fullWidth
                        sx={{ mt: 2 }}
                        disabled={isLoading}
                    />

                    {selectedFiles && (
                        <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
                            {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected
                        </Typography>
                    )}

                    {isLoading && (
                        <Box sx={{ mt: 2 }}>
                            <CircularProgress />
                            <Typography>Processing scan... This may take a few minutes.</Typography>
                        </Box>
                    )}

                    {error && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                            {error}
                        </Alert>
                    )}

                    {result && renderPredictionResult(result)}
                </Box>
            )}
        </Box>
    );
};

export default ImageTab;