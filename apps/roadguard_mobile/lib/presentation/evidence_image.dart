import 'package:flutter/material.dart';

import '../domain/road_defect_event.dart';
import '../services/evidence_file_stub.dart'
    if (dart.library.io) '../services/evidence_file_io.dart';

class EvidenceImage extends StatefulWidget {
  const EvidenceImage({super.key, required this.event, this.compact = false});

  final RoadDefectEvent event;
  final bool compact;

  @override
  State<EvidenceImage> createState() => _EvidenceImageState();
}

class _EvidenceImageState extends State<EvidenceImage> {
  Future<ImageProvider?>? _file;

  @override
  void initState() {
    super.initState();
    _loadFile();
  }

  @override
  void didUpdateWidget(EvidenceImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.event.evidenceLocalPath != oldWidget.event.evidenceLocalPath) {
      _loadFile();
    }
  }

  void _loadFile() {
    final path = widget.event.evidenceLocalPath;
    _file = path == null ? null : localEvidence(path);
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final event = widget.event;
    final placeholder = ColoredBox(
      color: colors.surfaceContainer,
      child: Center(
        child: widget.compact
            ? Icon(
                Icons.image_outlined,
                size: 21,
                color: colors.onSurfaceVariant,
              )
            : Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    Icons.hide_image_outlined,
                    size: 32,
                    color: colors.onSurfaceVariant,
                  ),
                  const SizedBox(height: 8),
                  const Text('الصورة غير متاحة'),
                ],
              ),
      ),
    );
    Widget image(ImageProvider provider, Widget fallback) => Image(
      image: provider,
      fit: BoxFit.cover,
      width: double.infinity,
      height: double.infinity,
      semanticLabel: widget.compact ? null : 'صورة مرجعية لسطح الطريق',
      excludeFromSemantics: widget.compact,
      errorBuilder: (_, _, _) => fallback,
    );

    final url = Uri.tryParse(event.evidenceImageUrl ?? '');
    final hasUrl =
        url != null &&
        url.hasAuthority &&
        {'https', 'http'}.contains(url.scheme) &&
        url.userInfo.isEmpty;
    final remote = hasUrl
        ? image(
            NetworkImage(url.toString(), headers: event.imageHeaders),
            placeholder,
          )
        : placeholder;
    final local = _file == null
        ? remote
        : FutureBuilder<ImageProvider?>(
            future: _file,
            builder: (_, snapshot) {
              final provider = snapshot.data;
              return provider == null ? remote : image(provider, remote);
            },
          );
    final asset = event.evidenceAsset;
    return ClipRRect(
      borderRadius: BorderRadius.circular(widget.compact ? 12 : 16),
      child: asset == null ? local : image(AssetImage(asset), local),
    );
  }
}
