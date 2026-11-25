import 'dart:io';

import 'package:flutter/painting.dart';
import 'package:path_provider/path_provider.dart';

final _evidencePath = RegExp(
  r'^evidence/[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)$',
);

Future<ImageProvider?> localEvidence(String relativePath) async {
  if (!_evidencePath.hasMatch(relativePath)) return null;
  try {
    final directory = await getApplicationDocumentsDirectory();
    final file = File('${directory.path}/$relativePath');
    if (await file.exists()) {
      return ResizeImage.resizeIfNeeded(1200, null, FileImage(file));
    }
  } on FileSystemException {
    return null;
  } on MissingPlatformDirectoryException {
    return null;
  }
  return null;
}
