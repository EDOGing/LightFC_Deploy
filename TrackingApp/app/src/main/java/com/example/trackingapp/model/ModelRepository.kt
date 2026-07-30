package com.example.trackingapp.model

import android.content.Context
import java.io.File

class ModelRepository(context: Context) {
    internal val rootDirectory = File(context.filesDir, "models").apply {
        check(isDirectory || mkdirs()) { "无法创建应用私有模型目录" }
    }

    fun listModels(): List<ModelInfo> =
        rootDirectory.listFiles()
            .orEmpty()
            .asSequence()
            .filter { it.isDirectory && !it.name.startsWith(".") && isSafeId(it.name) }
            .mapNotNull { directory -> runCatching { ModelInfo.fromDirectory(directory) }.getOrNull() }
            .sortedByDescending { it.importedAtMillis }
            .toList()

    fun deleteModel(modelId: String): Boolean {
        require(isSafeId(modelId)) { "非法模型 ID" }
        val directory = File(rootDirectory, modelId)
        require(directory.canonicalFile.parentFile == rootDirectory.canonicalFile) { "非法模型目录" }
        if (directory.exists()) {
            require(!ModelInfo.fromDirectory(directory).isBundled) { "内置模型不能删除" }
        }
        return !directory.exists() || directory.deleteRecursively()
    }

    internal fun nextAvailableId(modelName: String): String {
        val base = modelName
            .replace(Regex("[^A-Za-z0-9._-]+"), "_")
            .trim('.', '_', '-')
            .ifEmpty { "model" }
            .take(64)

        var candidate = base
        var suffix = 2
        while (File(rootDirectory, candidate).exists()) {
            candidate = "$base-$suffix"
            suffix += 1
        }
        return candidate
    }

    private fun isSafeId(value: String): Boolean =
        value.isNotBlank() && value.length <= 80 && SAFE_ID.matches(value)

    companion object {
        private val SAFE_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]*")
    }
}
