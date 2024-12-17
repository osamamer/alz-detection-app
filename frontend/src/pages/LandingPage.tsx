import React from "react";
import { Box, Typography, Button, Grid, Container, Card, CardContent, CardMedia } from "@mui/material";

interface LandingPageProps {
    darkMode: boolean;
    darkModeFunction: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ darkMode, darkModeFunction }) => {
    return (
        <Box
            sx={{
                backgroundColor: darkMode ? "#121212" : "#ffffff",
                color: darkMode ? "#ffffff" : "#000000",
                minHeight: "100vh", // Ensure the page takes full height and is scrollable
                paddingBottom: "2rem",
            }}
        >
            {/* Hero Section */}
            <Box
                sx={{
                    height: "100vh",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundImage: `url('https://via.placeholder.com/1920x1080')`,
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    textAlign: "center",
                    padding: "2rem",
                }}
            >
                <Container maxWidth="md">
                    <Typography variant="h2" gutterBottom>
                        Understanding Alzheimer’s Disease
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                        Discover the impact, symptoms, and causes of Alzheimer’s, and explore resources for caregiving and support.
                    </Typography>
                    <Button
                        variant="contained"
                        color="secondary"
                        size="large"
                        sx={{ marginTop: "1rem" }}
                    >
                        Learn More
                    </Button>
                </Container>
            </Box>

            {/* Section 1: What is Alzheimer’s */}
            <Container maxWidth="lg" sx={{ padding: "4rem 0" }}>
                <Grid container spacing={4}>
                    <Grid item xs={12} md={6}>
                        <Typography variant="h4" gutterBottom>
                            What is Alzheimer’s Disease?
                        </Typography>
                        <Typography variant="body1" color="text.secondary">
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus interdum lorem nec lacus
                            convallis, id vehicula purus ultricies.
                        </Typography>
                        <Button
                            variant="outlined"
                            color="primary"
                            sx={{ marginTop: "1rem" }}
                        >
                            Read More
                        </Button>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Card>
                            <CardMedia
                                component="img"
                                height="300"
                                image="https://via.placeholder.com/600x300"
                                alt="Alzheimer's Illustration"
                            />
                            <CardContent>
                                <Typography variant="caption" color="text.secondary">
                                    A brief caption or description for the image.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Container>

            {/* Section 2: Interactive Resources */}
            <Box sx={{ backgroundColor: darkMode ? "#333333" : "#f9f9f9", padding: "4rem 0" }}>
                <Container maxWidth="lg">
                    <Typography
                        variant="h4"
                        color="primary"
                        textAlign="center"
                        gutterBottom
                    >
                        Interactive Resources
                    </Typography>
                    <Grid container spacing={4}>
                        <Grid item xs={12} md={4}>
                            <Card>
                                <CardMedia
                                    component="img"
                                    height="200"
                                    image="https://via.placeholder.com/400x200"
                                    alt="Symptom Checker"
                                />
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Symptom Checker
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        A tool to help you identify early symptoms of Alzheimer’s.
                                    </Typography>
                                    <Button size="small" variant="contained" color="secondary">
                                        Try It
                                    </Button>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Card>
                                <CardMedia
                                    component="img"
                                    height="200"
                                    image="https://via.placeholder.com/400x200"
                                    alt="Caregiver Tips"
                                />
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Caregiver Tips
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Explore advice and best practices for Alzheimer’s caregiving.
                                    </Typography>
                                    <Button size="small" variant="contained" color="secondary">
                                        Learn More
                                    </Button>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Card>
                                <CardMedia
                                    component="img"
                                    height="200"
                                    image="https://via.placeholder.com/400x200"
                                    alt="Diagnosis Guide"
                                />
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Diagnosis Guide
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Understand the diagnostic process for Alzheimer’s disease.
                                    </Typography>
                                    <Button size="small" variant="contained" color="secondary">
                                        Get Started
                                    </Button>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </Container>
            </Box>

            {/* Section 3: Call to Action */}
            <Container maxWidth="md" sx={{ textAlign: "center", padding: "4rem 0" }}>
                <Typography variant="h4" gutterBottom>
                    Join the Fight Against Alzheimer’s
                </Typography>
                <Typography variant="body1" color="text.secondary" gutterBottom>
                    Support ongoing research, connect with caregivers, and spread awareness to create a brighter future for those affected by Alzheimer’s.
                </Typography>
                <Button variant="contained" color="primary" size="large">
                    Donate Now
                </Button>
                <Button
                    onClick={darkModeFunction}
                    variant="outlined"
                    color="secondary"
                    sx={{ marginLeft: "1rem" }}
                >
                    Toggle Theme
                </Button>
            </Container>
        </Box>
    );
};
