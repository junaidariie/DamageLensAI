import axios from "axios";

const BASE_URL =
  process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export const api = axios.create({ baseURL: BASE_URL });

export const predictComprehensive = (file, mode) => {
  const form = new FormData();
  form.append("file", file);
  return api.post(`/predict/comprehensive?mode=${mode}`, form, {
    timeout: 360000,
  });
};

export const getImageUrl = (path) => `${BASE_URL}${path}`;
