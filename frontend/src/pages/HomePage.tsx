import React, { useState } from 'react';
import { AppBar, Tabs, Tab, Box, Container, Typography } from '@mui/material';
import AudioTab from '../tabs/AudioTab';
import ImageTab from '../tabs/ImageTab';
import PictureTestTab from '../tabs/PictureTestTab';
import MemoryTestTab from '../tabs/MemoryTestTab';
import ClockDrawingTab from "../tabs/ClockDrawingTab";

export const HomePage = () => {
    const [tabIndex, setTabIndex] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabIndex(newValue);
    };

    return (
        <Container
            maxWidth="lg"
            sx={{
                py: 8,
                minHeight: '100vh', // Ensure the container takes the full viewport height
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* Header Section */}
            <Box textAlign="center" sx={{ mb: 6 }}>
                <Typography variant="h2">Alzheimer's Hub</Typography>
                <Typography variant="body1" sx={{ mt: 2 }}>
                    Choose an input type to analyze: Image, Audio, Picture Test, or Memory Test.
                </Typography>
            </Box>

            {/* Tabs Section */}
            <AppBar position="static" sx={{ borderRadius: 2, mb: 4 }}>
                <Tabs
                    value={tabIndex}
                    onChange={handleTabChange}
                    centered
                    sx={{
                        '& .MuiTabs-indicator': { backgroundColor: 'white' },
                        '& .MuiTab-root': { fontSize: '1.2rem', fontWeight: 'bold' },
                    }}
                >
                    <Tab label="Image" />
                    <Tab label="Audio" />
                    <Tab label="Picture Test" />
                    <Tab label="Memory Test" />
                    <Tab label="Clock Drawing Test" />
                </Tabs>
            </AppBar>

            {/* Tab Content Section */}
            <Box
                sx={{
                    // flex: 1, // Allow the Box to stretch to fill remaining height
                    // overflowY: 'auto', // Enable scrolling if content overflows
                }}
            >
                {tabIndex === 0 && <ImageTab />}
                {tabIndex === 1 && <AudioTab />}
                {tabIndex === 2 && <Box sx={{ border: '2px solid red' }}>
                    <PictureTestTab />
                </Box>}
                {tabIndex === 3 && <MemoryTestTab />}
                {tabIndex === 4 && <ClockDrawingTab />}
            </Box>
        </Container>
    );
};

export default HomePage;
