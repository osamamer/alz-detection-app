import React, { useState, useRef, useEffect } from 'react';
import {
    AppBar,
    Tabs,
    Tab,
    Box,
    Typography,
    Button,
    TextField,
    Card,
    CardContent,
    CardMedia,
    Container,
    Slider,
} from '@mui/material';
import PlayArrow from '@mui/icons-material/PlayArrow';
import Pause from '@mui/icons-material/Pause';

export function HomePage() {
    const [tabIndex, setTabIndex] = useState(0);
    const [imageType, setImageType] = useState<'2d' | '3d' | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [recording, setRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [waveformProgress, setWaveformProgress] = useState(0);
    const [recordingTime, setRecordingTime] = useState(0); // Time elapsed for recording

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    const MAX_RECORDING_TIME = 60; // Maximum recording time in seconds

    // Handle Tab Change
    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabIndex(newValue);
    };

    // Handle image upload
    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            const file = event.target.files[0];
            if (imageType === '2d') {
                const reader = new FileReader();
                reader.onload = () => setImagePreview(reader.result as string);
                reader.readAsDataURL(file);

                const formData = new FormData();
                formData.append('image', file);
                fetch('http://127.0.0.1:5000/predict-2d-image', { method: 'POST', body: formData });
            } else if (imageType === '3d') {
                const formData = new FormData();
                formData.append('image', file);
                fetch('http://127.0.0.1:5000/predict-3d-image', { method: 'POST', body: formData });
            }
        }
    };

    // Start recording
    const startRecording = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];

        mediaRecorderRef.current.ondataavailable = (event) => {
            audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.start();
        setRecording(true);
        setRecordingTime(0);
        setWaveformProgress(0);

        // Simulate waveform progress
        timerRef.current = setInterval(() => {
            setRecordingTime((prev) => {
                if (prev >= MAX_RECORDING_TIME) {
                    stopRecording();
                    return prev;
                }
                return prev + 1;
            });
            setWaveformProgress((prev) => Math.min(prev + 1, 100));
        }, 1000);
    };

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });

                // Generate a unique filename based on the current timestamp
                const timestamp = new Date().toISOString().replace(/[-:.]/g, '');
                const filename = `audio_clip_${timestamp}.wav`;

                setAudioBlob(blob);
                setRecording(false);
                clearInterval(timerRef.current!);

                // Send audio to backend
                const formData = new FormData();
                formData.append('audio', blob, filename); // Include the filename
                fetch('http://127.0.0.1:5000/predict-audio', { method: 'POST', body: formData });
            };
        }
    };

    // Toggle play/pause
    const togglePlayPause = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.onended = () => setIsPlaying(false);
        }
    }, [audioRef]);

    return (
        <Container maxWidth="lg" sx={{ py: 8 }}>
            {/* Header Section */}
            <Box textAlign="center" sx={{ mb: 6 }}>
                <Typography variant="h2" component="h1" gutterBottom>
                    Welcome to the AI Analysis Portal
                </Typography>
                <Typography variant="h5" color="text.secondary" gutterBottom>
                    Choose an input type to analyze: Image or Audio
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mt: 2, fontSize: '1.2rem' }}>
                    Use our advanced AI models to process images or audio for various applications. Select an option below to get started.
                </Typography>
            </Box>

            {/* Tab Navigation */}
            <AppBar position="static" sx={{ borderRadius: 2, mb: 4 }}>
                <Tabs
                    value={tabIndex}
                    onChange={handleTabChange}
                    centered
                    sx={{
                        '& .MuiTabs-indicator': { backgroundColor: 'white' },
                        '& .MuiTab-root': { fontSize: '1.5rem', fontWeight: 'bold' },
                    }}
                >
                    <Tab label="Image" />
                    <Tab label="Audio" />
                </Tabs>
            </AppBar>

            {/* Content Section */}
            <Box>
                {tabIndex === 0 && (
                    <Box>
                        <Typography variant="h4" align="center" sx={{ mb: 4 }}>
                            Image Processing
                        </Typography>
                        <Box textAlign="center">
                            <Button
                                variant="contained"
                                sx={{ mx: 3, px: 6, py: 2, fontSize: '1.2rem' }}
                                onClick={() => setImageType('2d')}
                            >
                                2D Image
                            </Button>
                            <Button
                                variant="contained"
                                sx={{ mx: 3, px: 6, py: 2, fontSize: '1.2rem' }}
                                onClick={() => setImageType('3d')}
                            >
                                3D Image
                            </Button>
                        </Box>

                        {imageType && (
                            <Box textAlign="center" sx={{ mt: 6 }}>
                                <Typography variant="h5" gutterBottom>
                                    Upload {imageType === '2d' ? '2D' : '3D'} Image
                                </Typography>
                                <TextField
                                    type="file"
                                    inputProps={{
                                        accept: imageType === '2d' ? 'image/*' : '.nii,.nii.gz',
                                    }}
                                    onChange={handleImageUpload}
                                    fullWidth
                                    sx={{ mt: 3 }}
                                />
                                {imageType === '2d' && imagePreview && (
                                    <Card
                                        sx={{
                                            mt: 6,
                                            mx: 'auto',
                                            maxWidth: 600,
                                            borderRadius: 4,
                                            boxShadow: 3,
                                        }}
                                    >
                                        <CardMedia
                                            component="img"
                                            alt="2D Preview"
                                            height="400"
                                            image={imagePreview}
                                        />
                                        <CardContent>
                                            <Typography variant="body1" color="text.secondary">
                                                Preview of the uploaded 2D image.
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                )}
                            </Box>
                        )}
                    </Box>
                )}

                {tabIndex === 1 && (
                    <Box textAlign="center">
                        <Typography variant="h4" sx={{ mb: 4 }}>
                            Audio Processing
                        </Typography>
                        {!recording && !audioBlob && (
                            <Button
                                variant="contained"
                                color="primary"
                                sx={{ py: 2, px: 4, fontSize: '1.2rem' }}
                                onClick={startRecording}
                            >
                                Start Recording
                            </Button>
                        )}

                        {recording && (
                            <Button
                                variant="contained"
                                color="secondary"
                                sx={{ py: 2, px: 4, fontSize: '1.2rem' }}
                                onClick={stopRecording}
                            >
                                Stop Recording
                            </Button>
                        )}

                        {!recording && audioBlob && (
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mt: 4,
                                }}
                            >
                                <Button
                                    variant="contained"
                                    color="primary"
                                    sx={{ minWidth: 60, minHeight: 60, borderRadius: '50%' }}
                                    onClick={togglePlayPause}
                                >
                                    {isPlaying ? <Pause /> : <PlayArrow />}
                                </Button>
                                <Box sx={{ mx: 4, flex: 1 }}>
                                    <Slider
                                        value={waveformProgress}
                                        onChange={() => {}}
                                        disabled
                                        sx={{
                                            color: 'primary.main',
                                            height: 8,
                                            borderRadius: 4,
                                            '& .MuiSlider-thumb': { display: 'none' },
                                        }}
                                    />
                                </Box>
                            </Box>
                        )}

                        {audioBlob && (
                            <audio
                                ref={audioRef}
                                src={URL.createObjectURL(audioBlob)}
                                style={{ display: 'none' }}
                            />
                        )}
                    </Box>
                )}
            </Box>
        </Container>
    );
}

export default HomePage;
