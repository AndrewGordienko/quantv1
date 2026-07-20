import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./styles.css";
import Layout from "./components/Layout.jsx";
import Today from "./pages/Today.jsx";
import Feed from "./pages/Feed.jsx";
import Leaderboard from "./pages/Leaderboard.jsx";
import Member from "./pages/Member.jsx";
import Ticker from "./pages/Ticker.jsx";
import Research from "./pages/Research.jsx";
import Evidence from "./pages/Evidence.jsx";
import PaperBook from "./pages/PaperBook.jsx";
import OpeningFlow from "./pages/OpeningFlow.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Today />} />
          <Route path="/feed" element={<Feed />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/member/:key" element={<Member />} />
          <Route path="/ticker/:ticker" element={<Ticker />} />
          <Route path="/research" element={<Research />} />
          <Route path="/paper" element={<PaperBook />} />
          <Route path="/opening-flow" element={<OpeningFlow />} />
          <Route path="/evidence" element={<Evidence />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
