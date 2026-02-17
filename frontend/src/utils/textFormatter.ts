/**
 * Utility functions for formatting text content
 */

/**
 * Formats explanation text by converting line breaks to proper HTML paragraphs
 * and cleaning up any markdown-style formatting
 */
export const formatExplanationText = (text: string): string => {
    if (!text) return "";

    // Clean up markdown formatting
    const cleanText = text
        .replace(/\*\*(.*?)\*\*/g, "$1") // Remove bold markdown
        .replace(/\*(.*?)\*/g, "$1") // Remove italic markdown
        .replace(/`(.*?)`/g, "$1") // Remove inline code markdown
        .trim();

    return cleanText;
};

/**
 * Formats feature list items with proper styling
 */
export const formatFeatureItem = (text: string): string => {
    return text
        .replace(/^\d+\.\s*/, "") // Remove list numbering
        .replace(/\*\*(.*?)\*\*/g, "$1") // Remove bold markdown
        .replace(/\*(.*?)\*/g, "$1") // Remove italic markdown
        .trim();
};
