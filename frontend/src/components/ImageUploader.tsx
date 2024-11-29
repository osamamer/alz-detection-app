import React, { useState } from "react";
import { Box, Button, Typography } from "@mui/material";
import axios from "axios";

const ImageUploader: React.FC = () => {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [file, setFile] = useState<File | null>(null);

    // Handle image upload and set preview URL
    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const uploadedFile = event.target.files?.[0];
        if (uploadedFile) {
            setFile(uploadedFile); // Save the file for backend submission
            setSelectedImage(URL.createObjectURL(uploadedFile)); // Generate a preview URL
        }
    };

    // Handle the image submission to the backend
    const handleImageSubmit = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await axios.post("http://localhost:5000/classify", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            console.log("Response from backend:", response.data);
            alert("Image successfully sent to the backend!");
        } catch (error) {
            console.error("Error submitting the image:", error);
            alert("Failed to send the image. Please try again.");
        }
    };

    return (
        <Box
            sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 2,
                marginTop: 4,
            }}
        >
            <Typography variant="h4" gutterBottom>
                Upload and Preview Image
            </Typography>
            <Button
                variant="contained"
                component="label"
                sx={{ textTransform: "none" }}
            >
                Choose Image
                <input
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={handleImageUpload}
                />
            </Button>
            {selectedImage && (
                <Box
                    sx={{
                        marginTop: 3,
                        border: "1px solid #ccc",
                        padding: 2,
                        borderRadius: 2,
                        boxShadow: 1,
                        width: "fit-content",
                    }}
                >
                    <img
                        src={selectedImage}
                        alt="Uploaded Preview"
                        style={{
                            maxWidth: "100%",
                            maxHeight: "300px",
                            borderRadius: "8px",
                        }}
                    />
                    <Button
                        variant="contained"
                        color="primary"
                        sx={{ marginTop: 2, textTransform: "none" }}
                        onClick={handleImageSubmit}
                    >
                        Confirm and Send
                    </Button>
                </Box>
            )}
        </Box>
    );
};

export default ImageUploader;
