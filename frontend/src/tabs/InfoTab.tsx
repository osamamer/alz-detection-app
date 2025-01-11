// InfoTab.tsx
import React from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Paper,
    Grid,
    Button,
    Link,
} from '@mui/material';
import {
    Warning,
    LocalHospital,
    Timeline,
    Psychology,
    Help,
    ArrowForward,
} from '@mui/icons-material';

const InfoTab = () => {
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
        <Box sx={{ p: 3 }}>
            {/* Introduction Section */}
            <Paper elevation={3} sx={{ p: 4, mb: 4, }}>
                <Typography variant="h4" gutterBottom color="primary">
                    Understanding Alzheimer's Disease
                </Typography>
                <Typography sx={{fontSize: '1.2rem'}} variant="body1" paragraph>
                    Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink and brain cells to die, leading to a gradual decline in memory, thinking, behavior, and social skills. It is the most common cause of dementia worldwide, affecting over 55 million people globally, with numbers expected to triple by 2050 as populations age. This devastating condition impacts not only those diagnosed but also their families and caregivers, creating ripple effects throughout communities and healthcare systems worldwide.
                </Typography>
            </Paper>

            {/* Warning Signs Section */}
            <Grid container spacing={4} sx={{ mb: 4 }}>
                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%' }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
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
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Card sx={{ height: '100%', }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <LocalHospital color="error" sx={{ mr: 1 }} />
                                <Typography variant="h5" color="error">
                                    When to Seek Help
                                </Typography>
                            </Box>
                            <Typography variant="body1" paragraph>
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
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Resources Section */}
            <Paper elevation={3} sx={{ p: 4, }}>
                <Typography variant="h5" gutterBottom color="primary">
                    Additional Resources
                </Typography>
                <Grid container spacing={3}>
                    {resourcesData.map((resource, index) => (
                        <Grid item xs={12} md={4} key={index}>
                            <Card sx={{ height: '100%' }}>
                                <CardContent>
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
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            </Paper>
        </Box>
    );
};

export default InfoTab;