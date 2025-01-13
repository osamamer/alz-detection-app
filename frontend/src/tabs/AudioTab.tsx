import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Typography,
    Button,
    Alert,
    Paper,
    Grid,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    SelectChangeEvent,
    CircularProgress
} from '@mui/material';
import { PlayArrow, Stop, CloudUpload } from '@mui/icons-material';

const AudioTab = () => {
    const [recording, setRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [recordingTime, setRecordingTime] = useState(0);
    const [age, setAge] = useState<string>('');
    const [gender, setGender] = useState<string>('');
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<{
        prediction: string;
        confidence: number;
        age: number;
        gender: string;
    } | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const MIN_RECORDING_TIME = 30; // 30 seconds
    const MAX_RECORDING_TIME = 60; // 60 seconds

    const handleAgeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.value;
        if (value === '' || (Number(value) >= 18 && Number(value) <= 120)) {
            setAge(value);
        }
    };

    const handleGenderChange = (event: SelectChangeEvent) => {
        setGender(event.target.value);
    };

    const validateInputs = (): boolean => {
        if (!age || Number(age) < 18 || Number(age) > 120) {
            setError('Please enter a valid age between 18 and 120');
            return false;
        }
        if (!gender) {
            setError('Please select a gender');
            return false;
        }
        return true;
    };

    const processAudioData = async (formData: FormData) => {
        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch('http://127.0.0.1:5000/predict-audio', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.error) {
                setError(data.error);
            } else {
                // Extract the nested result object
                const resultData = data.result;
                setResult({
                    prediction: resultData.prediction,
                    confidence: resultData.confidence,
                    age: resultData.age,
                    gender: resultData.gender
                });
            }
        } catch (err) {
            setError('Failed to process audio. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    // Start recording
    const startRecording = async () => {
        if (!validateInputs()) return;

        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };

            mediaRecorderRef.current.start();
            setRecording(true);
            setRecordingTime(0);

            timerRef.current = setInterval(() => {
                setRecordingTime(prev => {
                    if (prev >= MAX_RECORDING_TIME) {
                        stopRecording();
                        return MAX_RECORDING_TIME;
                    }
                    return prev + 1;
                });
            }, 1000);
        } catch (err) {
            setError('Could not access microphone. Please ensure microphone permissions are granted.');
        }
    };

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current && recording) {
            if (recordingTime < MIN_RECORDING_TIME) {
                setError(`Recording must be at least ${MIN_RECORDING_TIME} seconds long. Please try again.`);
                resetRecording();
                return;
            }

            mediaRecorderRef.current.stop();
            clearInterval(timerRef.current!);

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                const timestamp = new Date().toISOString().replace(/[-:.]/g, '');
                const filename = `audio_clip_${timestamp}.wav`;

                setAudioBlob(blob);
                setRecording(false);

                // Send to backend
                const formData = new FormData();
                formData.append('audio', blob, filename);
                formData.append('age', age);
                formData.append('gender', gender);

                processAudioData(formData);
            };
        }
    };

    const resetRecording = () => {
        setRecording(false);
        setAudioBlob(null);
        setRecordingTime(0);
        if (timerRef.current) {
            clearInterval(timerRef.current);
        }
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (!validateInputs()) return;

        const file = event.target.files?.[0];
        if (file) {
            // Check file type
            const acceptedTypes = ['audio/mp3', 'audio/wav', 'audio/mpeg', 'audio/wave', 'audio/x-wav'];
            if (!acceptedTypes.includes(file.type) &&
                !file.name.toLowerCase().endsWith('.mp3') &&
                !file.name.toLowerCase().endsWith('.wav')) {
                setError('Please upload an MP3 or WAV file.');
                return;
            }

            // Create AudioContext to check duration
            const audio = new Audio();
            audio.src = URL.createObjectURL(file);
            audio.onloadedmetadata = () => {
                if (audio.duration < MIN_RECORDING_TIME) {
                    setError(`Audio must be at least ${MIN_RECORDING_TIME} seconds long.`);
                    return;
                }
                // Process the file
                const formData = new FormData();
                formData.append('audio', file);
                formData.append('age', age);
                formData.append('gender', gender);

                processAudioData(formData);
            };
        }
    };

    return (
        <Box>
            {/* User Information */}
            <Paper elevation={3} sx={{ p: 4, mb: 4, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom color="primary">
                    Personal Information
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Age"
                            value={age}
                            onChange={(e) => {
                                const value = e.target.value.replace(/\D/g, '');
                                if (value === '' || (Number(value) >= 0 && Number(value) <= 120)) {
                                    setAge(value);
                                }
                            }}
                            inputProps={{
                                inputMode: 'numeric',
                                pattern: '[0-9]*'
                            }}
                            helperText="Must be between 18 and 120"
                        />
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Gender</InputLabel>
                            <Select
                                value={gender}
                                label="Gender"
                                onChange={handleGenderChange}
                            >
                                <MenuItem value="male">Male</MenuItem>
                                <MenuItem value="female">Female</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                </Grid>
            </Paper>

            {/* Audio Input Options */}
            <Grid container spacing={4} justifyContent="center" sx={{ mb: 4 }}>
                <Grid item xs={12} md={6}>
                    <Paper
                        elevation={3}
                        sx={{
                            p: 4,
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            borderRadius: 2
                        }}
                    >
                        <Button
                            variant="contained"
                            startIcon={recording ? <Stop /> : <PlayArrow />}
                            onClick={recording ? stopRecording : startRecording}
                            disabled={!!audioBlob}
                            sx={{
                                width: '100%',
                                py: 2,
                                fontSize: '1.1rem'
                            }}
                        >
                            {recording ? `Stop Recording (${recordingTime}s)` : 'Record Description'}
                        </Button>
                    </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper
                        elevation={3}
                        sx={{
                            p: 4,
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            borderRadius: 2
                        }}
                    >
                        <input
                            type="file"
                            accept=".mp3,.wav"
                            style={{ display: 'none' }}
                            ref={fileInputRef}
                            onChange={handleFileUpload}
                        />
                        <Button
                            variant="contained"
                            startIcon={<CloudUpload />}
                            onClick={() => fileInputRef.current?.click()}
                            sx={{
                                width: '100%',
                                py: 2,
                                fontSize: '1.1rem'
                            }}
                        >
                            Upload Audio File
                        </Button>
                    </Paper>
                </Grid>
            </Grid>

            {/* Error Messages */}
            {error && (
                <Alert severity="error" sx={{ mb: 4 }}>
                    {error}
                </Alert>
            )}

            {/* Loading Indicator */}
            {isLoading && (
                <Paper elevation={3} sx={{ p: 4, mb: 4, borderRadius: 2, bgcolor: 'background.paper' }}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                            <CircularProgress size={60} />
                        </Box>
                        <Typography variant="h6" color="primary">
                            Processing Audio...
                        </Typography>
                        <Typography variant="body2" color="text.secondary" textAlign="center">
                            This may take a few moments. Please don't close your browser.
                        </Typography>
                    </Box>
                </Paper>
            )}

            {/* Results Section */}
            {result && (
                <Paper
                    elevation={3}
                    sx={{
                        p: 4,
                        mb: 4,
                        borderRadius: 2,
                        // bgcolor: result.prediction === 'Control' ? 'success.light' : 'warning.light'
                    }}
                >
                    <Typography variant="h5" gutterBottom color="primary">
                        Analysis Results
                    </Typography>

                    <Box sx={{ mb: 3 }}>
                        <Typography variant="h6" gutterBottom sx={{color: result.prediction === 'Control' ? 'success.light' : 'warning.light'}}>
                            {result.prediction === 'Control'
                                ? 'No Cognitive Impairment Detected'
                                : 'Potential Cognitive Changes Detected'}
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2 }}>
                            {result.prediction === 'Control'
                                ? 'Based on your description, no signs of cognitive impairment were detected. Your language patterns are similar to those typically seen in healthy individuals.'
                                : 'Based on your description, some patterns in your language use suggest you may benefit from a cognitive health check-up with your healthcare provider.'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Confidence Level: {(result.confidence * 100).toFixed(1)}%
                        </Typography>
                    </Box>

                    <Typography variant="body2" color="text.secondary" paragraph>
                        Please note: This is not a medical diagnosis. This tool is designed to help identify potential signs that warrant further professional evaluation. For any concerns about cognitive health, please consult with a qualified healthcare provider.
                    </Typography>
                </Paper>
            )}

            {/* Cookie Theft Image */}
            <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
                <Typography variant="h5" gutterBottom color="primary">
                    Description Task
                </Typography>
                <Typography variant="body1" paragraph sx={{ mb: 3 }}>
                    Please describe everything you see happening in this picture in as much detail as possible.
                    Your description should be at least 30 seconds long.
                </Typography>
                <Box
                    component="img"
                    src="/cookie-theft.png"
                    alt="Cookie Theft Picture"
                    sx={{
                        width: '100%',
                        maxWidth: 800,
                        height: 'auto',
                        display: 'block',
                        margin: '0 auto',
                        borderRadius: 1
                    }}
                />
            </Paper>
        </Box>
    );
};

export default AudioTab;