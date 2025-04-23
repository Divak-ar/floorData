def extract_wall_coordinates(pred_mask, min_length=20):
    """
    Extract wall coordinates as line segments from prediction mask
    
    Returns:
        List of wall segments with start/end points and length
    """
    from skimage.morphology import skeletonize
    
    # Ensure binary mask
    binary = (pred_mask > 0).astype(np.uint8)
    
    # Clean the mask
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Get the centerlines
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    
    # Use Hough transform to get line segments
    lines = cv2.HoughLinesP(
        skeleton, 
        rho=1, 
        theta=np.pi/180, 
        threshold=10, 
        minLineLength=min_length, 
        maxLineGap=10
    )
    
    wall_segments = []
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            wall_segments.append({
                "id": i,
                "points": [
                    {"x": int(x1), "y": int(y1)},
                    {"x": int(x2), "y": int(y2)}
                ],
                "length": float(length)
            })
    
    return wall_segments