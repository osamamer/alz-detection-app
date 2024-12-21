import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, Slider } from '@mui/material';
import PlayArrow from '@mui/icons-material/PlayArrow';
import Pause from '@mui/icons-material/Pause';

const AudioTab = () => {
    const [recording, setRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [waveformProgress, setWaveformProgress] = useState(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    const MAX_RECORDING_TIME = 60;

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
        setWaveformProgress(0);

        timerRef.current = setInterval(() => {
            setWaveformProgress((prev) => Math.min(prev + 1, 100));
        }, 1000);
    };

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                const timestamp = new Date().toISOString().replace(/[-:.]/g, '');
                const filename = `audio_clip_${timestamp}.wav`;

                setAudioBlob(blob);
                setRecording(false);
                clearInterval(timerRef.current!);

                const formData = new FormData();
                formData.append('audio', blob, filename);
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
        <Box textAlign="center">
            <Typography variant="h4" gutterBottom>
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
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 4 }}>
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
    );
};

export default AudioTab;
