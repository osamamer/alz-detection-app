import {AppBar, Box, Button, IconButton, TextField, Toolbar, Typography} from "@mui/material";
import NightlightIcon from '@mui/icons-material/Nightlight';
import ChatIcon from '@mui/icons-material/Chat';
import {useNavigate} from "react-router-dom";
import React from "react";
import {TaskToCreate} from "../interfaces/TaskToCreate.tsx";

type props = {onSubmit: (taskToCreate: TaskToCreate) => void,
            darkModeFunction: (darkMode: boolean) => void,
            darkMode: boolean};

export function TopBar(props: props) {
    // const {toggleTheme, isDarkMode} = useContext(ThemeContext);
    const navigate = useNavigate();
    const handleFeedbackClick = () => {
        navigate('/feedback');
    }
    const handleLogoutClick = () => {
        navigate('/login');
    }
    return (
        <>
            <AppBar sx={{
                '& .MuiToolbar-root': {
                    paddingLeft: 0,
                },
                backgroundColor: "background.default",
            }} className="appbar">
                <Toolbar sx={{display: 'flex', padding: 0, justifyContent: 'space-between'}} className="appbar">
                    <Box sx={{
                        display: 'flex', alignItems: 'center', flexGrow: 1,
                        flexBasis: 0, gap: 4, marginBottom: 4
                    }}>
                        {/*<img src={logo} alt="logo" className="icon"/>*/}

                    </Box>

                    <Typography
                        variant="h6" component="div">
                        Alzheimer's Hub
                    </Typography>
                    <Box sx={{
                        display: 'flex', justifyContent: 'flex-end', gap: 1, flexGrow: 1,
                        flexBasis: 0
                    }}>
                        <IconButton onClick={() => {
                            props.darkModeFunction(props.darkMode)
                        }}>
                            <NightlightIcon></NightlightIcon>
                        </IconButton>
                        <Button
                            variant="outlined"
                            onClick={handleFeedbackClick}
                            startIcon={<ChatIcon/>}
                            title="Give us your feedback!"
                        >
                            Feedback
                        </Button>

                        <Button sx={{
                            '&:hover': {
                                color: theme => theme.palette.secondary.main
                            },
                        }} color="inherit" onClick={handleLogoutClick}>Log out</Button>
                    </Box>
                </Toolbar>
            </AppBar>
        </>
    )
}