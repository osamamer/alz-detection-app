import '../App.css'
import {Box} from "@mui/material";
import {TopBar} from "../components/TopBar.tsx";
import ImageUploader from "../components/ImageUploader";


type props = { darkMode: boolean, darkModeFunction: (darkMode: boolean) => void };

export function HomePage(props: props) {
    const ROOT_URL = "http://localhost:8080";


    return (
        <>
            <Box sx={{
                // width: 1,
                maxWidth: '100%',
                display: 'flex',
                overflow: 'hidden',
                justifyContent: 'center',
                mt: 8, flexGrow: 1, mr: 0,
                padding: 3,
            }}>
                <TopBar onSubmit={() => {}} darkMode={props.darkMode} darkModeFunction={props.darkModeFunction}/>
                <ImageUploader/>
            </Box>


        </>
    )
}

export default HomePage
