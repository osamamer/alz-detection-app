import React, { useState } from 'react';
import { Box, Typography, Button, TextField, List, ListItem } from '@mui/material';

export const MemoryTestTab = () => {
    const [showWords, setShowWords] = useState(false);
    const [userInput, setUserInput] = useState('');
    const [score, setScore] = useState<number | null>(null);
    const [submitted, setSubmitted] = useState(false);

    const wordList = ['apple', 'house', 'river', 'chair', 'dog']; // Word bank
    const correctWords = new Set(wordList);

    const handleStartTest = () => {
        setShowWords(true);
        setTimeout(() => {
            setShowWords(false);
        }, 5000); // Show words for 5 seconds
    };

    const handleSubmit = () => {
        const userWords = userInput.split(' ').map((word) => word.trim().toLowerCase());
        const correctCount = userWords.filter((word) => correctWords.has(word)).length;
        setScore(correctCount);
        setSubmitted(true);
    };

    return (
        <Box textAlign="center">
            <Typography variant="h4" gutterBottom>
                Memory Test
            </Typography>
            <Typography variant="body1" sx={{ mb: 4 }}>
                You will see a list of words for 5 seconds. Try to remember as many as you can and type them below.
            </Typography>

            {/* Display Words */}
            {showWords ? (
                <List sx={{ mb: 4, maxWidth: 400, mx: 'auto' }}>
                    {wordList.map((word, index) => (
                        <ListItem key={index}>
                            <Typography variant="h6">{word}</Typography>
                        </ListItem>
                    ))}
                </List>
            ) : (
                <Box sx={{ mb: 4 }}>
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={handleStartTest}
                        disabled={showWords}
                        sx={{ py: 2, px: 4, fontSize: '1.2rem' }}
                    >
                        Start Test
                    </Button>
                </Box>
            )}

            {/* Input Field */}
            {!showWords && !submitted && (
                <Box sx={{ maxWidth: 600, mx: 'auto' }}>
                    <TextField
                        label="Your Words"
                        placeholder="Enter words separated by whitespaces"
                        multiline
                        fullWidth
                        rows={4}
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        sx={{ mb: 3 }}
                    />
                    <Button
                        variant="contained"
                        color="secondary"
                        onClick={handleSubmit}
                        disabled={!userInput.trim()}
                        sx={{ px: 4, py: 2 }}
                    >
                        Submit
                    </Button>
                </Box>
            )}

            {/* Results */}
            {submitted && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6">Your Score: {score} / {wordList.length}</Typography>
                    <Typography variant="body2" color="text.secondary">
                        Words you remembered: {score}/{wordList.length}
                    </Typography>
                </Box>
            )}
        </Box>
    );
};

export default MemoryTestTab;
