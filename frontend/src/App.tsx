import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import {CssBaseline, ThemeProvider} from "@mui/material";
import {useState} from 'react';
import {HomePage} from "./pages/HomePage";
import {LandingPage} from "./pages/LandingPage";
import {darkTheme, lightTheme} from "./Theme";

function App() {
    const [darkMode, setDarkMode] = useState(true);
    const toggleTheme = () => {
        setDarkMode(!darkMode);
    };

    return (
        <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
            <CssBaseline />
            <Router>
                <Routes>
                    <Route path="/" element={<HomePage darkMode={darkMode} darkModeFunction={toggleTheme} />} />
                    <Route path="/landing" element={<LandingPage darkMode={darkMode} darkModeFunction={toggleTheme} />} />
                </Routes>
            </Router>
        </ThemeProvider>
    );
}

export default App;
