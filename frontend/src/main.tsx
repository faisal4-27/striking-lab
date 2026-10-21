import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { Layout } from "./components/ui";
import History from "./pages/History";
import Label from "./pages/Label";
import Record from "./pages/Record";
import Result from "./pages/Result";
import Upload from "./pages/Upload";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<History />} />
          <Route path="record" element={<Record />} />
          <Route path="upload" element={<Upload />} />
          <Route path="shots/:id" element={<Result />} />
          <Route path="label" element={<Label />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
