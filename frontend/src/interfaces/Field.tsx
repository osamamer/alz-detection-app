export interface Field {
    name: string;
    label: string;
    type?: string;
    placeholder?: string;
    required?: boolean;
    min?: number;
    max?: number;
}