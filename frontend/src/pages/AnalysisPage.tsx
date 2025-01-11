// AnalysisPage.tsx
import React, { useState } from 'react';
import {
    AppBar,
    Tabs,
    Tab,
    Box,
    Container,
    Typography,
    Button
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { ArrowBack } from '@mui/icons-material';
import AudioTab from '../tabs/AudioTab';
import ImageTab from '../tabs/ImageTab';

export const AnalysisPage = () => {
    const [tabIndex, setTabIndex] = useState(0);
    const navigate = useNavigate();

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabIndex(newValue);
    };

    return (
        <Container
            // maxWidth="md"
            sx={{
                py: 4,
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {/* Back Button */}
            <Button
                startIcon={<ArrowBack />}
                onClick={() => navigate('/')}
                sx={{ mb: 4, alignSelf: 'flex-start' }}
            >
                Back to Information
            </Button>

            {/* Header */}
            <Box textAlign="center" sx={{ mb: 4 }}>
                <Typography variant="h3" color="primary.main">
                    Analysis Tools
                </Typography>
            </Box>

            {/* Tabs Section */}
            <AppBar
                position="static"
                sx={{
                    borderRadius: 2,
                    mb: 4,
                }}
            >
                <Tabs
                    value={tabIndex}
                    onChange={handleTabChange}
                    centered
                    sx={{
                        '& .MuiTabs-indicator': { backgroundColor: 'primary.main' },
                        '& .MuiTab-root': {
                            fontSize: '1.2rem',
                            fontWeight: 'bold',
                            color: 'text.primary',
                            '&.Mui-selected': {
                                color: 'primary.main',
                            }
                        },
                    }}
                >
                    <Tab
                        icon={
                            <Box
                                component="img"
                                src="/2d-brain.png"
                                alt="MRI Analysis"
                                sx={{ height: 28, width: 28, mb: 1, filter: 'brightness(0) invert(1)'  }}
                            />
                        }
                        label="MRI Analysis"
                    />
                    <Tab
                        icon={
                            <Box
                                component="img"
                                src="/audio-icon.png"
                                alt="Audio Analysis"
                                sx={{ height: 28, width: 28, mb: 1, filter: 'brightness(0) invert(1)' }}
                            />
                        }
                        label="Audio Analysis"
                    />
                </Tabs>
            </AppBar>

            {/* Tab Content */}
            <Box sx={{ flex: 1 }}>
                {tabIndex === 0 && <ImageTab />}
                {tabIndex === 1 && <AudioTab />}
            </Box>
        </Container>
    );
};

export default AnalysisPage;