import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, Slider, Alert, Paper, Grid } from '@mui/material';
import { PlayArrow, Stop, CloudUpload } from '@mui/icons-material';

const AudioTab = () => {
    const [recording, setRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [recordingTime, setRecordingTime] = useState(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const MIN_RECORDING_TIME = 30; // 30 seconds
    const MAX_RECORDING_TIME = 60; // 60 seconds

    // Start recording
    const startRecording = async () => {
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
                fetch('http://127.0.0.1:5000/predict-audio', { method: 'POST', body: formData });
            };
        }
    };

    const resetRecording = () => {
        setRecording(false);
        setAudioBlob(null);
        setRecordingTime(0);
        clearInterval(timerRef.current!);
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            // Check file type
            if (!['audio/mp3', 'audio/wav'].includes(file.type)) {
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
                fetch('http://127.0.0.1:5000/predict-audio', { method: 'POST', body: formData });
            };
        }
    };

    return (
        <Box>
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

            {/* Cookie Theft Image */}
            <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
                <Typography variant="h5" gutterBottom color="primary">
                    Cookie Theft Picture Description Task
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