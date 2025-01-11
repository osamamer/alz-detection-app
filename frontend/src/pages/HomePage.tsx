// HomePage.tsx
import React from 'react';
import {
    Box,
    Container,
    Typography,
    Button,
    Grid,
    Paper,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Link,
} from '@mui/material';
import {
    Warning,
    LocalHospital,
    Psychology,
    Timeline,
    Help,
    ArrowForward,
    AssessmentRounded,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

export const HomePage = () => {
    const navigate = useNavigate();

    const warningSignsData = [
        "Memory loss that disrupts daily life",
        "Challenges in planning or solving problems",
        "Difficulty completing familiar tasks",
        "Confusion with time or place",
        "Problems with visual images and spatial relationships",
        "New problems with words in speaking or writing",
        "Misplacing things and losing the ability to retrace steps",
        "Decreased or poor judgment",
        "Withdrawal from work or social activities",
        "Changes in mood and personality"
    ];

    const resourcesData = [
        {
            title: "Alzheimer's Association",
            link: "https://www.alz.org",
            description: "24/7 Helpline: 800-272-3900"
        },
        {
            title: "National Institute on Aging",
            link: "https://www.nia.nih.gov/health/alzheimers",
            description: "Research and clinical trials information"
        },
        {
            title: "Mayo Clinic",
            link: "https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447",
            description: "Comprehensive medical information"
        }
    ];

    return (
        <Container
            sx={{
                py: 4,
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
                maxWidth: '100% !important',
            }}
        >
            {/* Header */}
            <Box textAlign="center" sx={{ mb: 4 }}>
                <Typography
                    variant="h2"
                    sx={{
                        fontWeight: 700,
                        color: 'primary.main',
                        mb: 1
                    }}
                >
                    Alzheimer's Hub
                </Typography>
            </Box>

            {/* Main Content Grid */}
            <Grid container spacing={4} sx={{ mt: 2 }}>
                {/* Information Section */}
                <Grid item xs={12} md={7}>
                    <Paper
                        elevation={3}
                        sx={{
                            p: 4,
                            height: '100%',
                            borderRadius: 2
                        }}
                    >
                        <Typography variant="h4" gutterBottom color="primary.main">
                            Understanding Alzheimer's Disease
                        </Typography>
                        <Typography style={{  textAlign: 'justify',
                            textJustify: 'inter-word'}} variant="body1" paragraph>
                            Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink and brain cells to die, leading to a gradual decline in memory, thinking, behavior, and social skills. It is the most common cause of dementia worldwide, affecting over 55 million people globally, with numbers expected to triple by 2050 as populations age. This devastating condition impacts not only those diagnosed but also their families and caregivers, creating ripple effects throughout communities and healthcare systems worldwide.
                        </Typography>
                        <Button
                            variant="outlined"
                            color="primary"
                            onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })}
                            sx={{ mt: 2 }}
                        >
                            Learn More
                        </Button>
                    </Paper>
                </Grid>

                {/* Action Section */}
                <Grid item xs={12} md={5}>
                    <Paper
                        elevation={3}
                        sx={{
                            p: 4,
                            height: '100%',
                            borderRadius: 2,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center',
                            alignItems: 'center',
                        }}
                    >
                        <Typography variant="h4" gutterBottom color="primary.main" textAlign="center">
                            Try Our Analysis Tools
                        </Typography>
                        <Typography variant="body1" paragraph textAlign="center">
                            Use our AI-powered tools to analyze MRI scans and audio recordings for early detection of Alzheimer's disease indicators.
                        </Typography>
                        <Button
                            variant="contained"
                            color="primary"
                            size="large"
                            onClick={() => navigate('/analysis')}
                            startIcon={<AssessmentRounded />}
                            sx={{
                                mt: 3,
                                px: 4,
                                py: 2,
                                fontSize: '1.1rem',
                            }}
                        >
                            Start Analysis
                        </Button>
                    </Paper>
                </Grid>
            </Grid>

            {/* Additional Information Sections */}
            <Box sx={{ mt: 6 }}>
                {/* Warning Signs Section */}
                <Grid container spacing={4} sx={{ mb: 4 }}>
                    <Grid item xs={12} md={6}>
                        <Paper sx={{ height: '100%', p: 4, borderRadius: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, borderRadius: 2
                            }}>
                                <Warning color="warning" sx={{ mr: 1 }} />
                                <Typography variant="h5" color="warning.main">
                                    10 Warning Signs
                                </Typography>
                            </Box>
                            <List>
                                {warningSignsData.map((sign, index) => (
                                    <ListItem key={index}>
                                        <ListItemIcon>
                                            <ArrowForward color="primary" fontSize="small" />
                                        </ListItemIcon>
                                        <ListItemText primary={sign} />
                                    </ListItem>
                                ))}
                            </List>
                        </Paper>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Paper sx={{ height: '100%', p: 4, borderRadius: 2}}>
                            <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2, borderRadius: 2 }}>
                                <LocalHospital color="error" sx={{ mr: 1 }} />
                                <Typography variant="h5" color="error">
                                    When to Seek Help
                                </Typography>
                            </Box>
                            <Typography variant="body1" paragraph textAlign="left">
                                If you or a loved one experiences any of these warning signs, it's important to:
                            </Typography>
                            <List>
                                <ListItem>
                                    <ListItemIcon>
                                        <Psychology color="primary" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Schedule an appointment with your primary care physician"
                                        secondary="They can perform initial assessments and provide referrals"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <Timeline color="primary" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Document changes in behavior and symptoms"
                                        secondary="Keep a journal of observations to share with healthcare providers"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <Help color="primary" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Don't delay seeking help"
                                        secondary="Early diagnosis allows for better treatment options and planning"
                                    />
                                </ListItem>
                            </List>
                        </Paper>
                    </Grid>
                </Grid>
                <Grid container spacing={4} sx={{ mb: 4 }}>
                    <Grid item xs={12} md={6}>
                        <Paper sx={{ height: '100%', p: 4, borderRadius: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, borderRadius: 2 }}>
                                <Timeline color="info" sx={{ mr: 1 }} />
                                <Typography variant="h5" color="info.main">
                                    Stages of Alzheimer's
                                </Typography>
                            </Box>
                            <List>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Early Stage (Mild)"
                                        secondary="Memory lapses, difficulty with planning and organization, misplacing objects"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Middle Stage (Moderate)"
                                        secondary="Increased confusion, poor judgment, behavioral changes, need for assistance with daily tasks"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Late Stage (Severe)"
                                        secondary="Loss of awareness, difficulty communicating, full-time care needed, physical symptoms"
                                    />
                                </ListItem>
                            </List>
                        </Paper>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Paper sx={{ height: '100%', p: 4, borderRadius: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, borderRadius: 2 }}>
                                <Psychology color="success" sx={{ mr: 1 }} />
                                <Typography variant="h5" color="success.main">
                                    Prevention & Health Tips
                                </Typography>
                            </Box>
                            <List>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Regular Physical Exercise"
                                        secondary="At least 150 minutes of moderate activity per week"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Mental Stimulation"
                                        secondary="Reading, puzzles, learning new skills, social engagement"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Healthy Lifestyle"
                                        secondary="Balanced diet, quality sleep, stress management, regular health check-ups"
                                    />
                                </ListItem>
                                <ListItem>
                                    <ListItemIcon>
                                        <ArrowForward color="primary" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary="Cardiovascular Health"
                                        secondary="Managing blood pressure, cholesterol, and diabetes"
                                    />
                                </ListItem>
                            </List>
                        </Paper>
                    </Grid>
                </Grid>
                {/* Resources Section */}
                <Box sx={{ py: 4, px: 0, width: '100%'}}>
                    <Typography variant="h5" gutterBottom color="primary">
                        Additional Resources
                    </Typography>
                    <Grid container spacing={3}>
                        {resourcesData.map((resource, index) => (
                            <Grid item xs={12} md={4} key={index}>
                                <Paper sx={{ height: '100%', p: 3, borderRadius: 2 }}>
                                    <Typography variant="h6" gutterBottom>
                                        {resource.title}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" paragraph>
                                        {resource.description}
                                    </Typography>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        href={resource.link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        Learn More
                                    </Button>
                                </Paper>
                            </Grid>
                        ))}
                    </Grid>
                </Box>
            </Box>
        </Container>
    );
};

export default HomePage;