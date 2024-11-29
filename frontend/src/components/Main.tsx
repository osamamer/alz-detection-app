import {Box} from "@mui/material";
type props =
export const Main = ({props}) => {
    return (
        <><Box sx={{
            // width: 1,
            // maxWidth: 1500,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            justifyContent: 'center',
            mt: 10, flexGrow: 1, mr: 0,
            padding: 3,
        }}>
            {children}
        </Box></>
    );
};