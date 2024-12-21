import React, { useState } from 'react';
import { AppBar, Tabs, Tab, Box, Container, Typography } from '@mui/material';
import AudioTab from '../tabs/AudioTab';
import ImageTab from '../tabs/ImageTab';
import PictureTestTab from '../tabs/PictureTestTab';
import MemoryTestTab from '../tabs/MemoryTestTab';
import ClockDrawingTab from "../tabs/ClockDrawingTab"; // Import the new Memory Test tab

export const HomePage = () => {
    const [tabIndex, setTabIndex] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabIndex(newValue);
    };

    return (
        <Container maxWidth="lg" sx={{ py: 8 }}>
            <Box textAlign="center" sx={{ mb: 6 }}>
                <Typography variant="h2">Alzheimer's Hub</Typography>
                <Typography variant="body1" sx={{ mt: 2 }}>
                    Choose an input type to analyze: Image, Audio, Picture Test, or Memory Test.
                </Typography>
            </Box>

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
                    <Tab label="Clock Drawing Test"/>
                </Tabs>
            </AppBar>

            <Box>
                {tabIndex === 0 && <ImageTab />}
                {tabIndex === 1 && <AudioTab />}
                {tabIndex === 2 && <PictureTestTab />}
                {tabIndex === 3 && <MemoryTestTab />} {/* Render the Memory Test */}
                {tabIndex === 4 && <ClockDrawingTab/>}
            </Box>
        </Container>
    );
};

export default HomePage;
